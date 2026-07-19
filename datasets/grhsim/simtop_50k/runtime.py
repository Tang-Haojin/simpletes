"""Trusted SimTop-50k runtime for the GrhSIM SimpleTES benchmark.

The performance result from this module is accepted only when topology,
quiet-window, affinity, ASLR, NUMA placement, scheduler, PMU, and functional
audits all pass.  Host load is deliberately reported as retryable
infrastructure rather than as a bad candidate.

The evaluator owns source isolation and builds.  This module consumes already
built artifacts, stages each run to fresh NUMA-first-touched tmpfs inodes, and
runs them.  Pure parsing/selection/command-building helpers are kept public so
the protocol can be tested without executing a simulation.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


EXPECTED_PERSONALITY = "00040000"
DEFAULT_PERF_EVENTS = (
    "cycles:u",
    "instructions:u",
    "de_no_dispatch_per_slot.no_ops_from_frontend:u",
    "cpu/de_no_dispatch_per_slot.no_ops_from_frontend,cmask=0x6/u",
    "de_no_dispatch_per_slot.backend_stalls:u",
    "task-clock",
    "context-switches",
    "cpu-migrations",
)
DEFAULT_PMU_EVENTS = (
    "cycles:u",
    "instructions:u",
    "de_no_dispatch_per_slot.no_ops_from_frontend:u",
    "cpu/de_no_dispatch_per_slot.no_ops_from_frontend,cmask=0x6/u",
    "de_no_dispatch_per_slot.backend_stalls:u",
)
NEGATIVE_LOG_RE = re.compile(
    r"mismatch|assert|fatal|error|fail|bad trap|segmentation|aborted|input_fullpass_blocked",
    re.I,
)


class TrustedRuntimeError(RuntimeError):
    """Base error carrying machine-readable diagnostics."""

    def __init__(self, message: str, *, diagnostics: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


class RetryableInfrastructureError(TrustedRuntimeError):
    """The candidate was not measured because the environment was untrusted."""


class InvalidCandidateError(TrustedRuntimeError):
    """The candidate ran in a trusted environment but failed its own gates."""


def parse_cpu_list(value: str) -> tuple[int, ...]:
    """Parse a Linux cpulist such as ``0-3,8,10-11``."""

    cpus: set[int] = set()
    text = value.strip()
    if not text:
        return ()
    for part in text.split(","):
        part = part.strip()
        if not part:
            raise ValueError(f"empty cpulist component in {value!r}")
        if "-" in part:
            first_text, last_text = part.split("-", 1)
            first, last = int(first_text), int(last_text)
            if first < 0 or last < first:
                raise ValueError(f"invalid cpulist range {part!r}")
            cpus.update(range(first, last + 1))
        else:
            cpu = int(part)
            if cpu < 0:
                raise ValueError(f"invalid CPU {cpu}")
            cpus.add(cpu)
    return tuple(sorted(cpus))


def format_cpu_list(cpus: Iterable[int]) -> str:
    """Format CPU IDs as a compact, deterministic Linux cpulist."""

    ordered = sorted(set(cpus))
    if not ordered:
        return ""
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


@dataclass(frozen=True)
class CpuTopology:
    cpu: int
    numa_node: int
    socket_id: int
    core_id: int
    siblings: tuple[int, ...]
    l3_cpus: tuple[int, ...]


@dataclass(frozen=True)
class CcdTopology:
    cpus: tuple[int, ...]
    numa_node: int
    socket_id: int
    physical_cores: tuple[tuple[int, ...], ...]

    @property
    def key(self) -> str:
        return f"node{self.numa_node}:{format_cpu_list(self.cpus)}"


@dataclass(frozen=True)
class GateAssessment:
    passed: bool
    count: int
    mean_idle: float
    min_idle: float
    target_idle: float | None = None
    sibling_idle: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class Placement:
    ccd: CcdTopology
    cpu: int
    sibling: int
    helper_cpu: int
    gate: GateAssessment

    def as_dict(self) -> dict[str, Any]:
        return {
            "ccd": self.ccd.key,
            "ccd_cpus": list(self.ccd.cpus),
            "numa_node": self.ccd.numa_node,
            "cpu": self.cpu,
            "sibling": self.sibling,
            "helper_cpu": self.helper_cpu,
            "gate": asdict(self.gate),
        }


@dataclass(frozen=True)
class ArtifactSet:
    name: str
    binary: Path
    image: Path
    nemu: Path
    repo: Path | None = None


@dataclass
class RuntimeConfig:
    """Configuration accepted directly or through :meth:`from_mapping`."""

    source_repo: Path | None = None
    slot_root: Path | None = None
    workspace: Path | None = None
    env_sh: Path | None = None
    results_dir: Path | None = None
    image: Path | None = None
    nemu: Path | None = None
    candidate_spec: Mapping[str, Any] = field(default_factory=dict)
    sysfs_root: Path = Path("/sys/devices/system/cpu")
    shm_root: Path = Path("/dev/shm")
    expected_ccd_logical_cpus: int = 16
    gate_seconds: int = 3
    gate_attempts: int = 3
    mean_idle_threshold: float = 98.0
    min_idle_threshold: float = 95.0
    target_idle_threshold: float = 98.0
    # Two mirrored pairs produce the formal A-B-B-A sequence by default.
    samples_per_variant: int = 2
    group_order: str = "ABBA"
    cycles: int = 50_000
    run_timeout_seconds: float = 600.0
    max_gate_to_run_gap_seconds: float = 5.0
    child_discovery_seconds: float = 10.0
    process_audit_delay_seconds: float = 5.0
    perf_events: tuple[str, ...] = DEFAULT_PERF_EVENTS
    pmu_events: tuple[str, ...] = DEFAULT_PMU_EVENTS
    min_perf_scheduled_percent: float = 99.9
    min_task_clock_scheduled_percent: float = 99.9
    min_cpus_utilized: float = 0.995
    max_context_switches_per_second: float = 20.0
    min_binary_pages: int = 20_000
    min_binary_local_ratio: float = 0.999
    min_nemu_pages: int = 100
    min_nemu_local_ratio: float = 0.95
    keep_staged: bool = False
    env_overrides: Mapping[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | "RuntimeConfig" | None) -> "RuntimeConfig":
        if isinstance(value, cls):
            return value
        raw = dict(value or {})
        if raw.get("workspace") is None and raw.get("slot_root") is not None:
            raw["workspace"] = raw["slot_root"]
        if raw.get("slot_root") is None and raw.get("workspace") is not None:
            raw["slot_root"] = raw["workspace"]
        known = {item.name for item in fields(cls)}
        filtered = {key: val for key, val in raw.items() if key in known}
        path_fields = {
            "source_repo",
            "slot_root",
            "workspace",
            "env_sh",
            "results_dir",
            "image",
            "nemu",
            "sysfs_root",
            "shm_root",
        }
        for key in path_fields:
            if key in filtered and filtered[key] is not None:
                filtered[key] = Path(filtered[key])
        for key in ("perf_events", "pmu_events"):
            if key in filtered:
                filtered[key] = tuple(filtered[key])
        config = cls(**filtered)
        if config.results_dir is None and config.slot_root is not None:
            config.results_dir = config.slot_root / "simtop_50k_results"
        return config


def _read_int(path: Path, default: int | None = None) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except FileNotFoundError:
        if default is not None:
            return default
        raise


def _cpu_number(cpu_dir: Path) -> int:
    match = re.fullmatch(r"cpu(\d+)", cpu_dir.name)
    if match is None:
        raise ValueError(f"not a CPU sysfs directory: {cpu_dir}")
    return int(match.group(1))


def _l3_shared_cpu_list(cpu_dir: Path) -> tuple[int, ...]:
    cache_root = cpu_dir / "cache"
    for index_dir in sorted(cache_root.glob("index*")):
        try:
            level = (index_dir / "level").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        if level == "3":
            return parse_cpu_list(
                (index_dir / "shared_cpu_list").read_text(encoding="utf-8")
            )
    # The benchmark machine exposes L3 as index3.  This fallback also makes a
    # minimally mocked sysfs tree useful without weakening production checks.
    fallback = cache_root / "index3" / "shared_cpu_list"
    if fallback.exists():
        return parse_cpu_list(fallback.read_text(encoding="utf-8"))
    raise ValueError(f"CPU {_cpu_number(cpu_dir)} has no L3 shared_cpu_list")


def _cpu_numa_node(cpu_dir: Path) -> int:
    nodes = []
    for path in cpu_dir.glob("node[0-9]*"):
        match = re.fullmatch(r"node(\d+)", path.name)
        if match:
            nodes.append(int(match.group(1)))
    if len(nodes) != 1:
        raise ValueError(f"{cpu_dir} has {len(nodes)} NUMA-node links, expected one")
    return nodes[0]


def discover_cpu_topology(sysfs_root: str | Path = "/sys/devices/system/cpu") -> tuple[CpuTopology, ...]:
    """Read online CPU, SMT, NUMA, socket, core, and L3 topology from sysfs."""

    root = Path(sysfs_root)
    records: list[CpuTopology] = []
    for cpu_dir in sorted(root.glob("cpu[0-9]*"), key=_cpu_number):
        cpu = _cpu_number(cpu_dir)
        online_file = cpu_dir / "online"
        if online_file.exists() and _read_int(online_file) != 1:
            continue
        records.append(
            CpuTopology(
                cpu=cpu,
                numa_node=_cpu_numa_node(cpu_dir),
                socket_id=_read_int(cpu_dir / "topology" / "physical_package_id"),
                core_id=_read_int(cpu_dir / "topology" / "core_id"),
                siblings=parse_cpu_list(
                    (cpu_dir / "topology" / "thread_siblings_list").read_text(
                        encoding="utf-8"
                    )
                ),
                l3_cpus=_l3_shared_cpu_list(cpu_dir),
            )
        )
    if not records:
        raise ValueError(f"no online CPUs discovered below {root}")
    return tuple(records)


def enumerate_ccds(
    records: Sequence[CpuTopology], *, expected_logical_cpus: int = 16
) -> tuple[CcdTopology, ...]:
    """Group online CPUs by their sysfs L3 sharing set (one CCD on this host)."""

    online = {record.cpu for record in records}
    by_l3: dict[tuple[int, ...], list[CpuTopology]] = {}
    for record in records:
        l3 = tuple(cpu for cpu in record.l3_cpus if cpu in online)
        by_l3.setdefault(l3, []).append(record)

    ccds: list[CcdTopology] = []
    for cpus, members in sorted(by_l3.items()):
        if len(cpus) != expected_logical_cpus or {item.cpu for item in members} != set(cpus):
            continue
        nodes = {item.numa_node for item in members}
        sockets = {item.socket_id for item in members}
        if len(nodes) != 1 or len(sockets) != 1:
            continue
        cores = tuple(
            sorted(
                {
                    tuple(sorted(cpu for cpu in item.siblings if cpu in cpus))
                    for item in members
                }
            )
        )
        if any(len(core) != 2 for core in cores) or len(cores) * 2 != len(cpus):
            continue
        ccds.append(
            CcdTopology(
                cpus=cpus,
                numa_node=next(iter(nodes)),
                socket_id=next(iter(sockets)),
                physical_cores=cores,
            )
        )
    return tuple(ccds)


def parse_mpstat_idle(output: str) -> dict[int, float]:
    """Parse per-CPU ``Average:`` rows from ``LC_ALL=C mpstat`` output."""

    idle: dict[int, float] = {}
    for line in output.splitlines():
        fields_ = line.split()
        if len(fields_) < 3 or fields_[0] != "Average:" or not fields_[1].isdigit():
            continue
        try:
            value = float(fields_[-1])
        except ValueError:
            continue
        idle[int(fields_[1])] = value
    return idle


def _best_core(ccd: CcdTopology, idle: Mapping[int, float]) -> tuple[int, int] | None:
    available: list[tuple[float, float, int, int]] = []
    for core in ccd.physical_cores:
        if len(core) != 2 or any(cpu not in idle for cpu in core):
            continue
        target, sibling = sorted(core)
        available.append(
            (min(idle[target], idle[sibling]), (idle[target] + idle[sibling]) / 2, -target, sibling)
        )
    if not available:
        return None
    _, _, negative_target, sibling = max(available)
    return -negative_target, sibling


def assess_ccd_gate(
    ccd: CcdTopology,
    idle: Mapping[int, float],
    *,
    target: int | None = None,
    sibling: int | None = None,
    expected_count: int = 16,
    mean_threshold: float = 98.0,
    min_threshold: float = 95.0,
    target_threshold: float = 98.0,
) -> GateAssessment:
    """Apply the strict three-second whole-CCD pre-gate."""

    present = {cpu: idle[cpu] for cpu in ccd.cpus if cpu in idle}
    count = len(present)
    mean_idle = statistics.fmean(present.values()) if present else 0.0
    min_idle = min(present.values()) if present else 0.0
    if target is None or sibling is None:
        selected = _best_core(ccd, idle)
        if selected is not None:
            target, sibling = selected
    target_idle = idle.get(target) if target is not None else None
    sibling_idle = idle.get(sibling) if sibling is not None else None

    reasons: list[str] = []
    if count != expected_count:
        reasons.append(f"count={count}, expected={expected_count}")
    if mean_idle < mean_threshold:
        reasons.append(f"mean_idle={mean_idle:.3f}<{mean_threshold:.3f}")
    if min_idle < min_threshold:
        reasons.append(f"min_idle={min_idle:.3f}<{min_threshold:.3f}")
    if target_idle is None or target_idle < target_threshold:
        reasons.append(f"target_idle={target_idle!r}<{target_threshold:.3f}")
    if sibling_idle is None or sibling_idle < target_threshold:
        reasons.append(f"sibling_idle={sibling_idle!r}<{target_threshold:.3f}")
    return GateAssessment(
        passed=not reasons,
        count=count,
        mean_idle=mean_idle,
        min_idle=min_idle,
        target_idle=target_idle,
        sibling_idle=sibling_idle,
        reason="; ".join(reasons),
    )


def select_helper_cpu(ccd: CcdTopology, records: Sequence[CpuTopology]) -> int:
    """Select an online helper outside the measured CCD, preferring another NUMA node."""

    candidates = [record for record in records if record.cpu not in ccd.cpus]
    if not candidates:
        raise ValueError(f"no helper CPU exists outside {ccd.key}")
    return min(candidates, key=lambda item: (item.numa_node == ccd.numa_node, item.cpu)).cpu


def select_placement(
    ccds: Sequence[CcdTopology],
    idle_by_ccd: Mapping[str, Mapping[int, float]],
    records: Sequence[CpuTopology],
    *,
    expected_count: int = 16,
    mean_threshold: float = 98.0,
    min_threshold: float = 95.0,
    target_threshold: float = 98.0,
) -> Placement:
    """Choose the quietest passing physical core across dynamically found CCDs."""

    passing: list[tuple[tuple[float, float, float, int], Placement]] = []
    for ccd in ccds:
        idle = idle_by_ccd.get(ccd.key, {})
        core = _best_core(ccd, idle)
        if core is None:
            continue
        target, sibling = core
        gate = assess_ccd_gate(
            ccd,
            idle,
            target=target,
            sibling=sibling,
            expected_count=expected_count,
            mean_threshold=mean_threshold,
            min_threshold=min_threshold,
            target_threshold=target_threshold,
        )
        if not gate.passed:
            continue
        placement = Placement(
            ccd=ccd,
            cpu=target,
            sibling=sibling,
            helper_cpu=select_helper_cpu(ccd, records),
            gate=gate,
        )
        rank = (
            min(gate.target_idle or 0.0, gate.sibling_idle or 0.0),
            ((gate.target_idle or 0.0) + (gate.sibling_idle or 0.0)) / 2,
            gate.min_idle,
            -target,
        )
        passing.append((rank, placement))
    if not passing:
        raise RetryableInfrastructureError(
            "no dynamically discovered CCD passed the strict quiet-window gate",
            diagnostics={
                "ccd_count": len(ccds),
                "gate": "3s whole-CCD count=16 mean>=98 min>=95 target+sibling>=98",
            },
        )
    return max(passing, key=lambda item: item[0])[1]


def assess_continuous_monitor(
    placement: Placement,
    idle: Mapping[int, float],
    *,
    mean_threshold: float = 98.0,
    min_threshold: float = 95.0,
    sibling_threshold: float = 98.0,
) -> GateAssessment:
    """Audit the other 15 CCD logical CPUs for the complete workload window."""

    expected = set(placement.ccd.cpus) - {placement.cpu}
    present = {cpu: idle[cpu] for cpu in expected if cpu in idle}
    count = len(present)
    mean_idle = statistics.fmean(present.values()) if present else 0.0
    min_idle = min(present.values()) if present else 0.0
    sibling_idle = present.get(placement.sibling)
    reasons: list[str] = []
    if set(present) != expected:
        reasons.append(f"count={count}, expected={len(expected)}")
    if mean_idle < mean_threshold:
        reasons.append(f"mean_idle={mean_idle:.3f}<{mean_threshold:.3f}")
    if min_idle < min_threshold:
        reasons.append(f"min_idle={min_idle:.3f}<{min_threshold:.3f}")
    if sibling_idle is None or sibling_idle < sibling_threshold:
        reasons.append(f"sibling_idle={sibling_idle!r}<{sibling_threshold:.3f}")
    return GateAssessment(
        passed=not reasons,
        count=count,
        mean_idle=mean_idle,
        min_idle=min_idle,
        target_idle=None,
        sibling_idle=sibling_idle,
        reason="; ".join(reasons),
    )


def build_mpstat_command(helper_cpu: int, cpus: Iterable[int], seconds: int) -> list[str]:
    return [
        "taskset",
        "-c",
        str(helper_cpu),
        "mpstat",
        "-P",
        format_cpu_list(cpus),
        "1",
        str(seconds),
    ]


def build_monitor_command(placement: Placement, duration_seconds: int) -> list[str]:
    return build_mpstat_command(
        placement.helper_cpu,
        (cpu for cpu in placement.ccd.cpus if cpu != placement.cpu),
        duration_seconds,
    )


def build_first_touch_command(source: str | Path, destination: str | Path, cpu: int, node: int) -> list[str]:
    """Construct the non-reflink NUMA first-touch copy command."""

    return [
        "taskset",
        "-c",
        str(cpu),
        "numactl",
        f"--physcpubind={cpu}",
        f"--membind={node}",
        "cp",
        "--reflink=never",
        str(source),
        str(destination),
    ]


def build_aslr_probe_command() -> list[str]:
    return ["setarch", "x86_64", "-R", "sh", "-c", "cat /proc/self/personality"]


def build_source_env_command(env_sh: str | Path) -> list[str]:
    script = 'set -a\nsource "$1" >/dev/null\nenv -0'
    return ["bash", "-c", script, "simpletes-source-env", str(env_sh)]


def build_workload_command(
    artifacts: ArtifactSet,
    placement: Placement,
    perf_csv: str | Path,
    *,
    cycles: int = 50_000,
    perf_events: Sequence[str] = DEFAULT_PERF_EVENTS,
) -> list[str]:
    """Build the exact trusted workload nesting, without a shell."""

    return [
        "taskset",
        "-c",
        str(placement.cpu),
        "numactl",
        f"--physcpubind={placement.cpu}",
        f"--membind={placement.ccd.numa_node}",
        "perf",
        "stat",
        "-x,",
        "-o",
        str(perf_csv),
        "-e",
        ",".join(perf_events),
        "--",
        "setarch",
        "x86_64",
        "-R",
        str(artifacts.binary),
        "-i",
        str(artifacts.image),
        "--diff",
        str(artifacts.nemu),
        "-b",
        "0",
        "-e",
        "0",
        "-C",
        str(cycles),
    ]


def load_sourced_environment(
    env_sh: str | Path,
    *,
    base_env: Mapping[str, str] | None = None,
    run: Callable[..., subprocess.CompletedProcess[bytes]] = subprocess.run,
) -> dict[str, str]:
    """Source ``env.sh`` once and capture the exported environment safely."""

    completed = run(
        build_source_env_command(env_sh),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=dict(base_env or os.environ),
    )
    if completed.returncode != 0:
        stderr = completed.stderr.decode(errors="replace") if completed.stderr else ""
        raise RetryableInfrastructureError(f"failed to source env.sh: {stderr.strip()}")
    output = completed.stdout if isinstance(completed.stdout, bytes) else completed.stdout.encode()
    environment = dict(base_env or os.environ)
    for entry in output.split(b"\0"):
        if not entry or b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        environment[key.decode(errors="surrogateescape")] = value.decode(errors="surrogateescape")
    return environment


def audit_personality(value: str, expected: str = EXPECTED_PERSONALITY) -> bool:
    return value.strip().lower() == expected.lower()


def parse_cpus_allowed_list(status_text: str) -> tuple[int, ...]:
    match = re.search(r"^Cpus_allowed_list:\s*(\S+)\s*$", status_text, re.M)
    return parse_cpu_list(match.group(1)) if match else ()


def audit_process_state(
    status_text: str,
    personality_text: str,
    expected_cpu: int,
    *,
    resolved_exe: str | Path | None = None,
    expected_exe: str | Path | None = None,
) -> tuple[bool, dict[str, Any]]:
    allowed = parse_cpus_allowed_list(status_text)
    affinity_ok = allowed == (expected_cpu,)
    personality_ok = audit_personality(personality_text)
    exe_ok = True
    if expected_exe is not None:
        exe_ok = resolved_exe is not None and Path(resolved_exe).resolve() == Path(expected_exe).resolve()
    diagnostics = {
        "cpus_allowed_list": list(allowed),
        "expected_cpu": expected_cpu,
        "affinity_ok": affinity_ok,
        "personality": personality_text.strip(),
        "personality_ok": personality_ok,
        "resolved_exe": str(resolved_exe) if resolved_exe is not None else None,
        "exe_ok": exe_ok,
    }
    return affinity_ok and personality_ok and exe_ok, diagnostics


def parse_numa_maps_file_pages(
    numa_maps_text: str, paths: Iterable[str | Path]
) -> dict[str, dict[int, int]]:
    requested = {str(Path(path)): {} for path in paths}
    for line in numa_maps_text.splitlines():
        file_match = re.search(r"(?:^|\s)file=(\S+)", line)
        if not file_match or file_match.group(1) not in requested:
            continue
        counters = requested[file_match.group(1)]
        for node_text, pages_text in re.findall(r"(?:^|\s)N(\d+)=(\d+)(?=\s|$)", line):
            node, pages = int(node_text), int(pages_text)
            counters[node] = counters.get(node, 0) + pages
    return requested


def audit_numa_maps(
    numa_maps_text: str,
    *,
    binary: str | Path,
    nemu: str | Path,
    target_node: int,
    min_binary_pages: int = 20_000,
    min_binary_local_ratio: float = 0.999,
    min_nemu_pages: int = 100,
    min_nemu_local_ratio: float = 0.95,
) -> tuple[bool, dict[str, Any]]:
    pages = parse_numa_maps_file_pages(numa_maps_text, (binary, nemu))
    diagnostics: dict[str, Any] = {}
    passed = True
    for kind, path, min_pages, min_ratio in (
        ("binary", binary, min_binary_pages, min_binary_local_ratio),
        ("nemu", nemu, min_nemu_pages, min_nemu_local_ratio),
    ):
        counts = pages[str(Path(path))]
        total = sum(counts.values())
        local = counts.get(target_node, 0)
        ratio = local / total if total else 0.0
        ok = total >= min_pages and ratio >= min_ratio
        diagnostics[kind] = {
            "path": str(path),
            "pages_by_node": counts,
            "total_pages": total,
            "local_pages": local,
            "local_ratio": ratio,
            "min_pages": min_pages,
            "min_local_ratio": min_ratio,
            "ok": ok,
        }
        passed = passed and ok
    return passed, diagnostics


@dataclass(frozen=True)
class PerfRecord:
    event: str
    value: float | None
    runtime_ns: float | None
    scheduled_percent: float | None
    metric_value: float | None
    fields: tuple[str, ...]


def _perf_number(value: str) -> float | None:
    text = value.strip().replace(" ", "")
    if not text or text.startswith("<"):
        return None
    match = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_perf_stat_csv(output: str) -> list[PerfRecord]:
    """Parse the stable fields used from ``perf stat -x,`` output."""

    records: list[PerfRecord] = []
    for line in output.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = tuple(field_.strip() for field_ in line.split(","))
        if len(parts) < 5:
            continue
        event_index = 2
        event = parts[event_index]
        runtime_index = 3
        percent_index = 4
        if event.startswith("cpu/") and len(parts) >= 6 and "=" in parts[3]:
            event = f"{event},{parts[3]}"
            runtime_index = 4
            percent_index = 5
        runtime = _perf_number(parts[runtime_index]) if len(parts) > runtime_index else None
        scheduled = _perf_number(parts[percent_index]) if len(parts) > percent_index else None
        metric = _perf_number(parts[percent_index + 1]) if len(parts) > percent_index + 1 else None
        records.append(
            PerfRecord(
                event=event,
                value=_perf_number(parts[0]),
                runtime_ns=runtime,
                scheduled_percent=scheduled,
                metric_value=metric,
                fields=parts,
            )
        )
    return records


def audit_perf_stat(
    output: str,
    *,
    pmu_events: Sequence[str] = DEFAULT_PMU_EVENTS,
    min_scheduled_percent: float = 99.9,
    min_task_clock_scheduled_percent: float | None = None,
    min_cpus_utilized: float = 0.995,
    max_context_switches_per_second: float = 20.0,
) -> tuple[bool, dict[str, Any]]:
    records = {record.event: record for record in parse_perf_stat_csv(output)}
    diagnostics: dict[str, Any] = {"events": {}}
    passed = True
    for event in pmu_events:
        record = records.get(event)
        ok = bool(
            record
            and record.value is not None
            and record.scheduled_percent is not None
            and record.scheduled_percent >= min_scheduled_percent
        )
        diagnostics["events"][event] = {
            "value": record.value if record else None,
            "scheduled_percent": record.scheduled_percent if record else None,
            "ok": ok,
        }
        passed = passed and ok

    task_clock = records.get("task-clock")
    task_clock_threshold = (
        min_scheduled_percent
        if min_task_clock_scheduled_percent is None
        else min_task_clock_scheduled_percent
    )
    task_clock_ok = bool(
        task_clock
        and task_clock.scheduled_percent is not None
        and task_clock.scheduled_percent >= task_clock_threshold
        and task_clock.metric_value is not None
        and task_clock.metric_value >= min_cpus_utilized
    )
    diagnostics["task_clock"] = {
        "scheduled_percent": task_clock.scheduled_percent if task_clock else None,
        "min_scheduled_percent": task_clock_threshold,
        "cpus_utilized": task_clock.metric_value if task_clock else None,
        "ok": task_clock_ok,
    }
    passed = passed and task_clock_ok

    context = records.get("context-switches")
    context_rate = None
    if context and context.value is not None and context.runtime_ns and context.runtime_ns > 0:
        context_rate = context.value / (context.runtime_ns / 1_000_000_000.0)
    context_ok = bool(
        context
        and context.scheduled_percent is not None
        and context.scheduled_percent >= min_scheduled_percent
        and context_rate is not None
        and context_rate <= max_context_switches_per_second
    )
    diagnostics["context_switches"] = {
        "value": context.value if context else None,
        "per_second": context_rate,
        "scheduled_percent": context.scheduled_percent if context else None,
        "ok": context_ok,
    }
    passed = passed and context_ok

    migrations = records.get("cpu-migrations")
    migrations_ok = bool(
        migrations
        and migrations.value == 0
        and migrations.scheduled_percent is not None
        and migrations.scheduled_percent >= min_scheduled_percent
    )
    diagnostics["cpu_migrations"] = {
        "value": migrations.value if migrations else None,
        "scheduled_percent": migrations.scheduled_percent if migrations else None,
        "ok": migrations_ok,
    }
    passed = passed and migrations_ok
    diagnostics["ok"] = passed
    return passed, diagnostics


def audit_function_log(output: str) -> tuple[bool, int | None, dict[str, Any]]:
    """Check the GrhSIM endpoint and extract exactly one positive walltime."""

    signature_ok = "Core-0 instrCnt = 73580, cycleCnt = 49996" in output
    guest_cycle_ok = "Guest cycle spent: 50001" in output
    terminal_pc_ok = bool(
        re.search(
            r"EXCEEDING CYCLE/INSTR LIMIT at pc\s*=\s*0x80001312", output
        )
    )
    negative_matches = sorted({match.group(0) for match in NEGATIVE_LOG_RE.finditer(output)})
    walltimes = [int(value) for value in re.findall(r"Host time spent: ([0-9]+)ms", output)]
    walltime_ok = len(walltimes) == 1 and walltimes[0] > 0
    passed = (
        signature_ok
        and guest_cycle_ok
        and terminal_pc_ok
        and not negative_matches
        and walltime_ok
    )
    diagnostics = {
        "signature_ok": signature_ok,
        "guest_cycle_ok": guest_cycle_ok,
        "terminal_pc_ok": terminal_pc_ok,
        "negative_matches": negative_matches,
        "walltime_count": len(walltimes),
        "walltime_ms": walltimes[0] if len(walltimes) == 1 else None,
        "walltime_ok": walltime_ok,
        "ok": passed,
    }
    return passed, walltimes[0] if walltime_ok else None, diagnostics


def _format_artifact_value(value: Any, repo: Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value)
    if repo is not None:
        text = text.format(repo=str(repo))
    path = Path(text)
    if not path.is_absolute() and repo is not None:
        path = repo / path
    return path


def resolve_artifacts(
    value: str | Path | Mapping[str, Any], config: RuntimeConfig, *, role: str
) -> ArtifactSet:
    """Resolve pre-built artifacts while rejecting cross-worktree binary links."""

    supplied: dict[str, Any]
    if isinstance(value, Mapping):
        supplied = dict(value)
    else:
        path = Path(value)
        supplied = {"repo": path} if path.is_dir() else {"binary": path}

    spec: Mapping[str, Any] = config.candidate_spec
    if role in spec and isinstance(spec[role], Mapping):
        role_spec = spec[role]
    else:
        role_spec = spec
    repo_value = supplied.get("repo", role_spec.get("repo", config.source_repo))
    repo = Path(repo_value).absolute() if repo_value is not None else None
    binary_value = supplied.get("binary", supplied.get("emu", role_spec.get("binary")))
    if binary_value is None and repo is not None:
        binary_value = "build/xs/grhsim/grhsim-compile/emu"
    image_value = supplied.get("image", role_spec.get("image", config.image))
    nemu_value = supplied.get(
        "nemu", supplied.get("reference", role_spec.get("nemu", config.nemu))
    )
    binary = _format_artifact_value(binary_value, repo)
    image = _format_artifact_value(image_value, repo)
    nemu = _format_artifact_value(nemu_value, repo)
    missing_names = [
        name for name, artifact in (("binary", binary), ("image", image), ("nemu", nemu)) if artifact is None
    ]
    if missing_names:
        raise InvalidCandidateError(
            f"{role} artifact specification is missing {', '.join(missing_names)}"
        )

    assert binary is not None and image is not None and nemu is not None
    resolved: dict[str, Path] = {}
    repo_resolved = repo.resolve() if repo is not None else None
    for name, artifact in (("binary", binary), ("image", image), ("nemu", nemu)):
        if not artifact.is_file():
            raise InvalidCandidateError(f"{role} {name} does not exist: {artifact}")
        resolved_artifact = artifact.resolve()
        if not resolved_artifact.is_file():
            raise InvalidCandidateError(
                f"{role} {name} does not resolve to a regular file: {artifact}"
            )
        if repo_resolved is not None:
            try:
                resolved_artifact.relative_to(repo_resolved)
            except ValueError as error:
                raise InvalidCandidateError(
                    f"{role} {name} resolves outside its isolated repo: "
                    f"{artifact} -> {resolved_artifact}"
                ) from error
        resolved[name] = resolved_artifact
    resolved_binary = resolved["binary"]
    if not os.access(resolved_binary, os.X_OK):
        raise InvalidCandidateError(f"{role} binary is not executable: {resolved_binary}")
    return ArtifactSet(
        name=str(supplied.get("name", role)),
        binary=resolved_binary,
        image=resolved["image"],
        nemu=resolved["nemu"],
        repo=repo_resolved,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _retry_result(message: str, diagnostics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "valid": False,
        "infrastructure_retry": True,
        "retryable_infra": True,
        "error": message,
        "control_walltime_ms": None,
        "candidate_walltime_ms": None,
        "samples": [],
        "diagnostics": {"error": message, **dict(diagnostics or {})},
    }


class GrhSimRuntime:
    """Run one fixed-CCD/fixed-core control/candidate measurement group."""

    def __init__(
        self,
        config: RuntimeConfig | Mapping[str, Any] | None = None,
        *,
        run_command: Callable[..., subprocess.CompletedProcess[Any]] = subprocess.run,
        popen: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    ):
        self.config = RuntimeConfig.from_mapping(config)
        self._run_command = run_command
        self._popen = popen
        self._environment: dict[str, str] | None = None
        self._placement: Placement | None = None
        self._topology: tuple[CpuTopology, ...] = ()

    def evaluate(
        self,
        candidate: str | Path | Mapping[str, Any],
        control: str | Path | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate without leaking expected infrastructure failures as exceptions."""

        try:
            return self._evaluate_impl(candidate, control)
        except RetryableInfrastructureError as error:
            return _retry_result(str(error), error.diagnostics)
        except InvalidCandidateError as error:
            return {
                "valid": False,
                "infrastructure_retry": False,
                "retryable_infra": False,
                "error": str(error),
                "control_walltime_ms": None,
                "candidate_walltime_ms": None,
                "samples": [],
                "diagnostics": {"error": str(error), **error.diagnostics},
            }
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            return _retry_result(f"trusted runtime infrastructure error: {type(error).__name__}: {error}")

    def _prepare_environment(self) -> dict[str, str]:
        if self._environment is not None:
            return self._environment
        if self.config.env_sh is None or not self.config.env_sh.is_file():
            raise RetryableInfrastructureError(f"env.sh is missing: {self.config.env_sh}")
        environment = load_sourced_environment(
            self.config.env_sh, run=self._run_command
        )
        environment.update({str(key): str(value) for key, value in self.config.env_overrides.items()})
        environment["LC_ALL"] = "C"
        environment["EMU_PROGRESS_EVERY_CYCLES"] = "0"
        environment.pop("EMU_RUNTIME_PROFILE", None)
        for name in tuple(environment):
            if name.startswith("GRHSIM_TRACE_") or (
                name.startswith("WOLVRIX_GRHSIM_") and name.endswith("_TSV")
            ):
                environment.pop(name, None)
        self._environment = environment
        return environment

    def _require_tools(self, environment: Mapping[str, str]) -> None:
        path = environment.get("PATH")
        missing = [
            tool
            for tool in ("taskset", "numactl", "perf", "setarch", "mpstat", "cp")
            if shutil.which(tool, path=path) is None
        ]
        if missing:
            raise RetryableInfrastructureError(f"missing runtime tools: {', '.join(missing)}")

    @staticmethod
    def _pin_runner(cpu: int) -> set[int]:
        """Pin Python-side orchestration outside the measured CCD."""

        try:
            original = set(os.sched_getaffinity(0))
            if cpu not in original:
                raise RetryableInfrastructureError(
                    f"helper CPU {cpu} is outside the evaluator cpuset",
                    diagnostics={"allowed_cpus": sorted(original)},
                )
            os.sched_setaffinity(0, {cpu})
            if set(os.sched_getaffinity(0)) != {cpu}:
                raise RetryableInfrastructureError(
                    f"failed to pin runtime orchestrator to helper CPU {cpu}"
                )
            return original
        except OSError as error:
            raise RetryableInfrastructureError(
                f"failed to pin runtime orchestrator to helper CPU {cpu}: {error}"
            ) from error

    @staticmethod
    def _restore_runner_affinity(cpus: set[int]) -> None:
        if not cpus:
            return
        try:
            os.sched_setaffinity(0, cpus)
        except OSError as error:
            raise RetryableInfrastructureError(
                f"failed to restore runtime orchestrator affinity: {error}"
            ) from error

    def _run_mpstat(self, command: Sequence[str], environment: Mapping[str, str]) -> str:
        completed = self._run_command(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=dict(environment),
        )
        if completed.returncode != 0:
            raise RetryableInfrastructureError(
                f"mpstat failed with exit {completed.returncode}",
                diagnostics={"command": list(command), "stderr": completed.stderr},
            )
        return completed.stdout

    def _select_fixed_placement(self, environment: Mapping[str, str]) -> Placement:
        if self._placement is not None:
            return self._placement
        try:
            records = discover_cpu_topology(self.config.sysfs_root)
            allowed = set(os.sched_getaffinity(0))
            records = tuple(record for record in records if record.cpu in allowed)
            ccds = enumerate_ccds(
                records, expected_logical_cpus=self.config.expected_ccd_logical_cpus
            )
        except (OSError, ValueError) as error:
            raise RetryableInfrastructureError(f"failed to discover CPU/CCD topology: {error}") from error
        if not ccds:
            raise RetryableInfrastructureError(
                "no complete 16-logical-CPU CCD was discovered from L3 sysfs topology"
            )
        idle_by_ccd: dict[str, Mapping[int, float]] = {}
        for ccd in ccds:
            helper = select_helper_cpu(ccd, records)
            command = build_mpstat_command(helper, ccd.cpus, self.config.gate_seconds)
            original_affinity = self._pin_runner(helper)
            try:
                idle_by_ccd[ccd.key] = parse_mpstat_idle(
                    self._run_mpstat(command, environment)
                )
            finally:
                self._restore_runner_affinity(original_affinity)
        placement = select_placement(
            ccds,
            idle_by_ccd,
            records,
            expected_count=self.config.expected_ccd_logical_cpus,
            mean_threshold=self.config.mean_idle_threshold,
            min_threshold=self.config.min_idle_threshold,
            target_threshold=self.config.target_idle_threshold,
        )
        self._topology = records
        self._placement = placement
        return placement

    def _pre_gate(self, placement: Placement, environment: Mapping[str, str]) -> GateAssessment:
        last: GateAssessment | None = None
        for _ in range(self.config.gate_attempts):
            output = self._run_mpstat(
                build_mpstat_command(
                    placement.helper_cpu, placement.ccd.cpus, self.config.gate_seconds
                ),
                environment,
            )
            last = assess_ccd_gate(
                placement.ccd,
                parse_mpstat_idle(output),
                target=placement.cpu,
                sibling=placement.sibling,
                expected_count=self.config.expected_ccd_logical_cpus,
                mean_threshold=self.config.mean_idle_threshold,
                min_threshold=self.config.min_idle_threshold,
                target_threshold=self.config.target_idle_threshold,
            )
            if last.passed:
                return last
        raise RetryableInfrastructureError(
            "external load prevented the fixed CCD pre-gate",
            diagnostics={"placement": placement.as_dict(), "last_gate": asdict(last) if last else None},
        )

    def _stage_artifacts(
        self, artifacts: ArtifactSet, placement: Placement, environment: Mapping[str, str]
    ) -> tuple[ArtifactSet, Path, dict[str, Any]]:
        root = Path(
            tempfile.mkdtemp(
                prefix=f"simpletes-grhsim-{artifacts.name}-", dir=self.config.shm_root
            )
        )
        sources = {
            "binary": artifacts.binary,
            "image": artifacts.image,
            "nemu": artifacts.nemu,
        }
        names = {"binary": "emu", "image": "coremark.bin", "nemu": "nemu.so"}
        staged: dict[str, Path] = {}
        manifest: dict[str, Any] = {
            "copy_cpu": placement.cpu,
            "numa_node": placement.ccd.numa_node,
            "files": {},
        }
        try:
            for kind, source in sources.items():
                destination = root / names[kind]
                temporary = root / f".{names[kind]}.tmp-{uuid.uuid4().hex}"
                command = build_first_touch_command(
                    source, temporary, placement.cpu, placement.ccd.numa_node
                )
                completed = self._run_command(
                    command,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=dict(environment),
                )
                if completed.returncode != 0:
                    raise RetryableInfrastructureError(
                        f"NUMA first-touch staging failed for {kind}",
                        diagnostics={"command": command, "stderr": completed.stderr},
                    )
                os.replace(temporary, destination)
                source_stat = source.stat()
                destination_stat = destination.stat()
                source_sha = _sha256(source)
                destination_sha = _sha256(destination)
                if source_sha != destination_sha:
                    raise RetryableInfrastructureError(f"staged {kind} SHA-256 mismatch")
                if (source_stat.st_dev, source_stat.st_ino) == (
                    destination_stat.st_dev,
                    destination_stat.st_ino,
                ):
                    raise RetryableInfrastructureError(f"staged {kind} reused its source inode")
                staged[kind] = destination
                manifest["files"][kind] = {
                    "source": str(source),
                    "destination": str(destination),
                    "bytes": destination_stat.st_size,
                    "device": destination_stat.st_dev,
                    "inode": destination_stat.st_ino,
                    "sha256": destination_sha,
                    "command": command,
                }
            inode_keys = {
                (entry["device"], entry["inode"]) for entry in manifest["files"].values()
            }
            if len(inode_keys) != 3:
                raise RetryableInfrastructureError("staged inputs did not receive independent inodes")
            return (
                ArtifactSet(
                    name=artifacts.name,
                    binary=staged["binary"],
                    image=staged["image"],
                    nemu=staged["nemu"],
                    repo=artifacts.repo,
                ),
                root,
                manifest,
            )
        except Exception:
            shutil.rmtree(root, ignore_errors=True)
            raise

    @staticmethod
    def _descendant_pids(root_pid: int) -> set[int]:
        parents: dict[int, list[int]] = {}
        for stat_path in Path("/proc").glob("[0-9]*/stat"):
            try:
                stat = stat_path.read_text(encoding="utf-8")
                # The comm field may contain spaces and parentheses; ppid is
                # the second token after its final closing parenthesis.
                remainder = stat.rsplit(")", 1)[1].split()
                pid = int(stat_path.parent.name)
                ppid = int(remainder[1])
            except (OSError, ValueError, IndexError):
                continue
            parents.setdefault(ppid, []).append(pid)
        descendants: set[int] = set()
        frontier = [root_pid]
        while frontier:
            parent = frontier.pop()
            for child in parents.get(parent, ()):
                if child not in descendants:
                    descendants.add(child)
                    frontier.append(child)
        return descendants

    def _find_emu_pid(self, perf_pid: int, expected_exe: Path) -> int | None:
        deadline = time.monotonic() + self.config.child_discovery_seconds
        expected = expected_exe.resolve()
        while time.monotonic() < deadline:
            for pid in self._descendant_pids(perf_pid):
                try:
                    if (Path("/proc") / str(pid) / "exe").resolve() == expected:
                        return pid
                except OSError:
                    continue
            time.sleep(0.05)
        return None

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def _run_one(
        self,
        artifacts: ArtifactSet,
        placement: Placement,
        environment: Mapping[str, str],
        *,
        role: str,
        index: int,
    ) -> dict[str, Any]:
        staged, stage_root, manifest = self._stage_artifacts(artifacts, placement, environment)
        results_root = self.config.results_dir or Path(tempfile.gettempdir()) / "simtop_50k_results"
        results_root.mkdir(parents=True, exist_ok=True)
        stem = f"{index:02d}-{role}-{uuid.uuid4().hex[:8]}"
        perf_path = results_root / f"{stem}.perf.csv"
        emu_log_path = results_root / f"{stem}.emu.log"
        monitor_path = results_root / f"{stem}.monitor.log"
        audit_path = results_root / f"{stem}.audit.txt"
        monitor: subprocess.Popen[Any] | None = None
        workload: subprocess.Popen[Any] | None = None
        try:
            # Staging touches tens of thousands of pages and is intentionally
            # completed before the quiet gate.  Once the gate passes, permit
            # only the tiny ASLR probe/monitor setup before starting the run.
            pre_gate = self._pre_gate(placement, environment)
            gate_end_ns = time.monotonic_ns()
            probe = self._run_command(
                build_aslr_probe_command(),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(environment),
            )
            if probe.returncode != 0 or not audit_personality(probe.stdout):
                raise RetryableInfrastructureError(
                    "setarch -R personality probe failed",
                    diagnostics={"stdout": probe.stdout, "stderr": probe.stderr},
                )

            monitor_stream = monitor_path.open("w", encoding="utf-8")
            emu_stream = emu_log_path.open("w", encoding="utf-8")
            try:
                monitor = self._popen(
                    build_monitor_command(
                        placement, math.ceil(self.config.run_timeout_seconds) + 30
                    ),
                    stdout=monitor_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=dict(environment),
                    start_new_session=True,
                )
                run_start_ns = time.monotonic_ns()
                gate_to_run_gap_ms = (run_start_ns - gate_end_ns) / 1_000_000.0
                if gate_to_run_gap_ms > self.config.max_gate_to_run_gap_seconds * 1000:
                    raise RetryableInfrastructureError(
                        "quiet gate-to-run gap exceeded the trusted limit",
                        diagnostics={"gate_to_run_gap_ms": gate_to_run_gap_ms},
                    )
                workload = self._popen(
                    build_workload_command(
                        staged,
                        placement,
                        perf_path,
                        cycles=self.config.cycles,
                        perf_events=self.config.perf_events,
                    ),
                    stdout=emu_stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=dict(environment),
                    start_new_session=True,
                )
                emu_pid = self._find_emu_pid(workload.pid, staged.binary)
                if emu_pid is None:
                    return_code = workload.poll()
                    if return_code not in (None, 0):
                        raise InvalidCandidateError(
                            f"{role} exited before the trusted process audit (exit {return_code})"
                        )
                    raise RetryableInfrastructureError("could not locate the live emulator child")

                audit_deadline = time.monotonic() + self.config.process_audit_delay_seconds
                while time.monotonic() < audit_deadline and workload.poll() is None:
                    time.sleep(min(0.05, max(0.0, audit_deadline - time.monotonic())))
                proc_root = Path("/proc") / str(emu_pid)
                try:
                    status_text = (proc_root / "status").read_text(encoding="utf-8")
                    personality_text = (proc_root / "personality").read_text(encoding="utf-8")
                    numa_maps_text = (proc_root / "numa_maps").read_text(encoding="utf-8")
                    resolved_exe = (proc_root / "exe").resolve()
                except OSError as error:
                    if workload.poll() not in (None, 0):
                        raise InvalidCandidateError(
                            f"{role} exited before its audit completed: {error}"
                        ) from error
                    raise RetryableInfrastructureError(
                        f"failed to read live process audit: {error}"
                    ) from error
                process_ok, process_audit = audit_process_state(
                    status_text,
                    personality_text,
                    placement.cpu,
                    resolved_exe=resolved_exe,
                    expected_exe=staged.binary,
                )
                numa_ok, numa_audit = audit_numa_maps(
                    numa_maps_text,
                    binary=staged.binary,
                    nemu=staged.nemu,
                    target_node=placement.ccd.numa_node,
                    min_binary_pages=self.config.min_binary_pages,
                    min_binary_local_ratio=self.config.min_binary_local_ratio,
                    min_nemu_pages=self.config.min_nemu_pages,
                    min_nemu_local_ratio=self.config.min_nemu_local_ratio,
                )
                audit_path.write_text(
                    f"process={process_audit!r}\nnuma={numa_audit!r}\n",
                    encoding="utf-8",
                )
                if not process_ok:
                    raise RetryableInfrastructureError(
                        "affinity, executable, or fixed-ASLR process audit failed",
                        diagnostics={"process_audit": process_audit},
                    )
                if not numa_ok:
                    raise RetryableInfrastructureError(
                        "NUMA file-page locality audit failed",
                        diagnostics={"numa_audit": numa_audit},
                    )
                try:
                    return_code = workload.wait(timeout=self.config.run_timeout_seconds)
                except subprocess.TimeoutExpired as error:
                    raise InvalidCandidateError(
                        f"{role} timed out after {self.config.run_timeout_seconds}s"
                    ) from error
            finally:
                emu_stream.close()
                if workload is not None and workload.poll() is None:
                    self._terminate_process_group(workload)
                if monitor is not None and monitor.poll() is None:
                    try:
                        os.killpg(monitor.pid, signal.SIGINT)
                        monitor.wait(timeout=5)
                    except (OSError, subprocess.TimeoutExpired):
                        self._terminate_process_group(monitor)
                monitor_stream.close()

            if return_code != 0:
                raise InvalidCandidateError(f"{role} emulator exited with status {return_code}")
            monitor_text = monitor_path.read_text(encoding="utf-8")
            monitor_gate = assess_continuous_monitor(
                placement,
                parse_mpstat_idle(monitor_text),
                mean_threshold=self.config.mean_idle_threshold,
                min_threshold=self.config.min_idle_threshold,
                sibling_threshold=self.config.target_idle_threshold,
            )
            if not monitor_gate.passed:
                raise RetryableInfrastructureError(
                    "external load contaminated the fixed CCD during the workload",
                    diagnostics={"monitor_gate": asdict(monitor_gate)},
                )
            if not perf_path.is_file():
                raise RetryableInfrastructureError("perf stat did not produce its CSV")
            perf_ok, perf_audit = audit_perf_stat(
                perf_path.read_text(encoding="utf-8"),
                pmu_events=self.config.pmu_events,
                min_scheduled_percent=self.config.min_perf_scheduled_percent,
                min_task_clock_scheduled_percent=self.config.min_task_clock_scheduled_percent,
                min_cpus_utilized=self.config.min_cpus_utilized,
                max_context_switches_per_second=self.config.max_context_switches_per_second,
            )
            if not perf_ok:
                raise RetryableInfrastructureError(
                    "PMU scheduling, task-clock, context-switch, or migration audit failed",
                    diagnostics={"perf_audit": perf_audit},
                )
            emu_log = emu_log_path.read_text(encoding="utf-8")
            function_ok, walltime_ms, function_audit = audit_function_log(emu_log)
            if not function_ok or walltime_ms is None:
                raise InvalidCandidateError(
                    f"{role} failed the functional/unique-walltime gate",
                    diagnostics={"function_audit": function_audit},
                )
            return {
                "role": role,
                "index": index,
                "walltime_ms": walltime_ms,
                "pre_gate": asdict(pre_gate),
                "gate_to_run_gap_ms": gate_to_run_gap_ms,
                "monitor_gate": asdict(monitor_gate),
                "process_audit": process_audit,
                "numa_audit": numa_audit,
                "perf_audit": perf_audit,
                "function_audit": function_audit,
                "stage_manifest": manifest,
                "artifacts": {
                    "perf_csv": str(perf_path),
                    "emu_log": str(emu_log_path),
                    "monitor_log": str(monitor_path),
                    "audit": str(audit_path),
                },
            }
        finally:
            if workload is not None and workload.poll() is None:
                self._terminate_process_group(workload)
            if monitor is not None and monitor.poll() is None:
                self._terminate_process_group(monitor)
            if not self.config.keep_staged:
                shutil.rmtree(stage_root, ignore_errors=True)

    def _evaluate_impl(
        self,
        candidate: str | Path | Mapping[str, Any],
        control: str | Path | Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        environment = self._prepare_environment()
        self._require_tools(environment)
        candidate_artifacts = resolve_artifacts(candidate, self.config, role="candidate")
        control_artifacts = (
            resolve_artifacts(control, self.config, role="control") if control is not None else None
        )
        placement = self._select_fixed_placement(environment)
        original_affinity = self._pin_runner(placement.helper_cpu)
        try:
            schedule: list[tuple[str, ArtifactSet]] = []
            schedule_order: str
            if control_artifacts is None:
                schedule_order = "B" * self.config.samples_per_variant
                schedule = [
                    ("candidate", candidate_artifacts)
                    for _ in range(self.config.samples_per_variant)
                ]
            else:
                order = self.config.group_order.upper()
                if self.config.samples_per_variant != 2 or order not in {"ABBA", "BAAB"}:
                    raise ValueError(
                        "formal paired measurement requires samples_per_variant=2 and "
                        "group_order=ABBA or BAAB"
                    )
                schedule_order = order
                artifacts_by_role = {"A": control_artifacts, "B": candidate_artifacts}
                names_by_role = {"A": "control", "B": "candidate"}
                schedule = [
                    (names_by_role[role], artifacts_by_role[role]) for role in order
                ]

            samples: list[dict[str, Any]] = []
            for index, (role, artifacts) in enumerate(schedule, 1):
                try:
                    samples.append(
                        self._run_one(
                            artifacts, placement, environment, role=role, index=index
                        )
                    )
                except InvalidCandidateError as error:
                    if role == "control":
                        raise RetryableInfrastructureError(
                            f"trusted control failed: {error}", diagnostics=error.diagnostics
                        ) from error
                    raise
        finally:
            self._restore_runner_affinity(original_affinity)

        candidate_times = [sample["walltime_ms"] for sample in samples if sample["role"] == "candidate"]
        control_times = [sample["walltime_ms"] for sample in samples if sample["role"] == "control"]
        candidate_walltime = float(statistics.mean(candidate_times))
        control_walltime = float(statistics.mean(control_times)) if control_times else None
        speedup = control_walltime / candidate_walltime if control_walltime is not None else None
        improvement = (
            (control_walltime - candidate_walltime) / control_walltime
            if control_walltime is not None
            else None
        )
        return {
            "valid": True,
            "infrastructure_retry": False,
            "retryable_infra": False,
            "control_walltime_ms": control_walltime,
            "candidate_walltime_ms": candidate_walltime,
            "speedup": speedup,
            "relative_improvement": improvement,
            "samples": samples,
            "diagnostics": {
                "placement": placement.as_dict(),
                "group_fixed_ccd_cpu": True,
                "group_order": schedule_order,
                "headline_metric": "Host time spent walltime_ms",
                "aggregation": "arithmetic_mean",
                "candidate_walltime_range_ms": [min(candidate_times), max(candidate_times)],
                "control_walltime_range_ms": (
                    [min(control_times), max(control_times)] if control_times else None
                ),
            },
        }


def evaluate_candidate(
    candidate: str | Path | Mapping[str, Any],
    config: RuntimeConfig | Mapping[str, Any] | None = None,
    control: str | Path | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluator-friendly public entry point."""

    return GrhSimRuntime(config).evaluate(candidate, control=control)


__all__ = [
    "ArtifactSet",
    "CcdTopology",
    "CpuTopology",
    "DEFAULT_PERF_EVENTS",
    "DEFAULT_PMU_EVENTS",
    "EXPECTED_PERSONALITY",
    "GateAssessment",
    "GrhSimRuntime",
    "InvalidCandidateError",
    "PerfRecord",
    "Placement",
    "RetryableInfrastructureError",
    "RuntimeConfig",
    "assess_ccd_gate",
    "assess_continuous_monitor",
    "audit_function_log",
    "audit_numa_maps",
    "audit_perf_stat",
    "audit_personality",
    "audit_process_state",
    "build_aslr_probe_command",
    "build_first_touch_command",
    "build_monitor_command",
    "build_mpstat_command",
    "build_source_env_command",
    "build_workload_command",
    "discover_cpu_topology",
    "enumerate_ccds",
    "evaluate_candidate",
    "format_cpu_list",
    "load_sourced_environment",
    "parse_cpu_list",
    "parse_cpus_allowed_list",
    "parse_mpstat_idle",
    "parse_numa_maps_file_pages",
    "parse_perf_stat_csv",
    "resolve_artifacts",
    "select_helper_cpu",
    "select_placement",
]
