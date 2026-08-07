"""Repository-level evaluator for GrhSIM SimTop 50k optimization.

The evolved program is a versioned JSON document containing a unified diff and
an explicit attribution mode.  This module validates that document, applies the
diff to an evaluator-owned clone pinned to the benchmark revision, builds the
mode-appropriate current-default or explicitly-enabled variant, and delegates
only the machine-sensitive runtime protocol to :mod:`runtime`.

The user's checkout is a read-only source of Git objects and untracked
``env.sh`` configuration.  It is never reset, cleaned, patched, or built by this
evaluator.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass, replace
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import statistics
import subprocess
import sys
import time
from typing import Any, Callable, Iterator, Mapping, Sequence
import uuid


TASK_ROOT = Path(__file__).resolve().parent
SIMPLETES_ROOT = TASK_ROOT.parents[2]
DEFAULT_SOURCE_REPO = SIMPLETES_ROOT.parent / "wolvrix-playground-gsim-calibrate-5"
DEFAULT_SLOT_ROOT = Path("/tmp/simpletes-grhsim-simtop-50k")

PINNED_PARENT_COMMIT = "52ba7d9edcd713cd0ee3d8a605f1d4aa31b3c730"
PINNED_WOLVRIX_COMMIT = "d3ed9dea975bddf01185dde5c548a69241a09de9"
SCHEMA_VERSION = 2

MAX_CANDIDATE_BYTES = 2 * 1024 * 1024
MAX_PATCH_BYTES = 1024 * 1024
MAX_PATCH_FILES = 64
MAX_ENABLE_OPTIONS = 64
DEFAULT_BUILD_TIMEOUT_SECONDS = 4 * 60 * 60
DEFAULT_INFRA_RETRIES = 2
CONTROL_MARKER_SCHEMA_VERSION = 2
CANDIDATE_PROOF_SCHEMA_VERSION = 2
CANDIDATE_PROOF_VERSION = 2
RUNTIME_RESULT_SCHEMA_VERSION = 2
EVALUATION_RESULT_SCHEMA_VERSION = 2
EVALUATION_ATTEMPT_SCHEMA_VERSION = 2
CONTROL_CANDIDATE_PROOF_ID = "control"
CONTROL_CANDIDATE_PROOF_SHA256 = "0" * 64

_MARKER_START_RE = re.compile(r"(?m)^\s*(?:#|//)?\s*EVOLVE-BLOCK-START\s*$")
_MARKER_END_RE = re.compile(r"(?m)^\s*(?:#|//)?\s*EVOLVE-BLOCK-END\s*$")
_OPTION_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
_OPTION_STRING_RE = re.compile(r"^[A-Za-z0-9_.:+-]{1,256}$")
_MEASUREMENT_VALUE_RE = re.compile(
    r"(?:^|[_.:+-])(?:probe|profile|report|stats?|measure(?:ment)?|trace|dump|export)"
    r"(?:$|[_.:+-])",
    re.I,
)
_DIFF_HEADER_RE = re.compile(r"^diff --git a/([^\s]+) b/([^\s]+)$")
_INDEX_RE = re.compile(r"^index [0-9a-f]+\.\.[0-9a-f]+(?: ([0-9]{6}))?$")
_HOST_TIME_RE = re.compile(r"Host time spent:\s*([0-9]+)\s*ms", re.I)

_ALLOWED_PREFIXES = (
    "lib/",
    "include/",
    "app/pybind/",
)
_ALLOWED_SOURCE_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cxx",
    ".h",
    ".hh",
    ".hpp",
    ".hxx",
    ".inc",
}

# This is deliberately a closed list copied from the pinned playground's
# ``scripts/wolvrix_xs_grhsim.py`` generation boundary.  Diagnostic/profile,
# resume, stop, export, and measurement controls are intentionally absent.
# A candidate cannot expand this boundary because parent-repository scripts are
# outside the patch allowlist.
_ALLOWED_ENABLE_OPTIONS = frozenset(
    {
        "active_mask_gap_pack_policy",
        "commit_exact_event_policy",
        "commit_guard_event_buckets",
        "declared_value_compute_node_boundary",
        "deferred_activation_forward_policy",
        "direct_single_writer_state_reads",
        "dp_segment_penalty_ppm",
        "emit_parallelism",
        "enable_local_shared_compute",
        "enable_mem_to_reg",
        "final_fanin_pullback_max_moved_op_ppm",
        "final_fanin_pullback_max_moves",
        "final_fanin_pullback_max_node_ops",
        "final_fanin_pullback_max_value_width",
        "final_fanin_pullback_min_gain",
        "final_fanin_pullback_policy",
        "final_shared_input_peer_max_candidates",
        "final_shared_input_peer_max_inputs",
        "final_shared_input_peer_max_moved_op_ppm",
        "final_shared_input_peer_max_moves",
        "final_shared_input_peer_max_node_ops",
        "final_shared_input_peer_max_outputs",
        "final_shared_input_peer_max_peers",
        "final_shared_input_peer_max_value_width",
        "final_shared_input_peer_policy",
        "final_sibling_fusion_max_fused_op_ppm",
        "final_sibling_fusion_max_pairs",
        "final_sibling_fusion_min_gain",
        "final_sibling_fusion_policy",
        "final_terminal_pushforward_max_inputs",
        "final_terminal_pushforward_max_moved_op_ppm",
        "final_terminal_pushforward_max_moves",
        "final_terminal_pushforward_max_node_ops",
        "final_terminal_pushforward_max_outputs",
        "final_terminal_pushforward_max_value_width",
        "final_terminal_pushforward_min_bae_gain",
        "final_terminal_pushforward_min_boundary_value_gain",
        "final_terminal_pushforward_policy",
        "final_topo_policy",
        "full_active_word_consume",
        "kahn_level_pack_max_moved_op_ppm",
        "kahn_level_pack_max_moves",
        "kahn_level_pack_max_regression_ppm",
        "kahn_level_pack_policy",
        "local_shared_compute_common_owner_max_cloned_op_ppm",
        "local_shared_compute_common_owner_max_clones",
        "local_shared_compute_common_owner_policy",
        "local_shared_compute_max_cloned_op_ppm",
        "local_shared_compute_max_clones",
        "local_shared_compute_max_fanout",
        "local_shared_compute_max_width",
        "max_op_in_commit_supernode",
        "max_op_in_compute_node",
        "max_op_in_compute_supernode",
        "mem_to_reg_row_limit",
        "post_dp_refine_max_moved_op_ppm",
        "post_dp_refine_max_moves",
        "post_dp_refine_max_regression_ppm",
        "post_dp_refine_max_rounds",
        "post_dp_refine_policy",
        "pure_event_compute_word_bypass",
        "pure_event_word_pack_max_changed_word_ppm",
        "pure_event_word_pack_max_moved_supernode_ppm",
        "pure_event_word_pack_policy",
        "reg_to_mem_decoded_write_storage",
        "reg_to_mem_intent",
        "reg_to_mem_ordered_writes",
        "same_batch_activation_cohort_policy",
        "sched_batches_per_cpp",
        "sched_batch_max_estimated_lines",
        "sched_batch_max_ops",
        "sched_batch_target_count",
        "simplify_keep_declared_symbols",
        "skip_comb_lane_pack",
        "split_oversize_compute_nodes",
        "split_oversize_compute_node_max_ops",
    }
)

_BASE_FIXED_MAKE_ASSIGNMENTS = (
    "XS_NUM_CORES=1",
    "XS_EMU_THREADS=2",
    "XS_WITH_CHISELDB=0",
    "XS_WITH_CONSTANTIN=0",
    "XS_WAVEFORM=0",
    "XS_WOLF_GRHSIM_RESUME_FROM_PRE_REG_TO_MEM_JSON=0",
    "XS_WOLF_GRHSIM_RESUME_FROM_STATS_JSON=0",
    "XS_WOLF_GRHSIM_ENABLE_STATS=0",
    "WOLVRIX_GRHSIM_WAVEFORM=0",
    "WOLVRIX_GRHSIM_PERF=0",
)
_FORBIDDEN_PATCH_LINES = (
    "GIT binary patch",
    "Binary files ",
    "Subproject commit ",
    "rename from ",
    "rename to ",
    "copy from ",
    "copy to ",
    "similarity index ",
    "dissimilarity index ",
    "old mode ",
    "new mode ",
)

_SECRET_ENV_FRAGMENT_RE = re.compile(
    r"(?:API[_-]?KEY|APIKEY|TOKEN|PASSWORD|SECRET|CREDENTIAL|WEBHOOK|COOKIE|PRIVATE[_-]?KEY)",
    re.I,
)
_SECRET_TEXT_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s\"']+"),
    re.compile(
        r"(?i)((?:api[_-]?key|auth[_-]?token|access[_-]?token|password|secret)"
        r"\s*[=:]\s*[\"']?)[^\s,\"']+"
    ),
)

_ROOT_REQUIRED_SUBMODULES = {"wolvrix", "testcase/xiangshan"}
_GENERATION_INPUT_ROOTS: tuple[tuple[Path, frozenset[str] | None], ...] = (
    (Path("build/xs/rtl/rtl"), frozenset({".sv", ".v"})),
    (Path("testcase/xiangshan/build/generated-src"), None),
)
_REQUIRED_GENERATION_INPUTS = frozenset(
    {
        "build/xs/rtl/rtl/SimTop.sv",
        "testcase/xiangshan/build/generated-src/DifftestMacros.svh",
    }
)
_SCRUBBED_ENV_PREFIXES = (
    "WOLVRIX_XS_GRHSIM_",
    "WOLVRIX_GRHSIM_",
    "GRHSIM_",
)
_SCRUBBED_ENV_NAMES = {
    "EMU_RUNTIME_PROFILE",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
}


class CandidateError(ValueError):
    """Candidate document or patch failed a deterministic gate."""


class InfrastructureError(RuntimeError):
    """Evaluator workspace/build infrastructure failed."""


@dataclass(frozen=True)
class Candidate:
    schema_version: int
    candidate_mode: str
    hypothesis: str
    evidence: tuple[str, ...]
    patch: str
    enable_options: dict[str, bool | int | float | str]

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @property
    def is_control(self) -> bool:
        return self.candidate_mode == "control"

    @property
    def is_default_path(self) -> bool:
        return self.candidate_mode == "default-path"

    @property
    def is_explicit_options(self) -> bool:
        return self.candidate_mode == "explicit-options"

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "schema_version": self.schema_version,
                "candidate_mode": self.candidate_mode,
                "hypothesis": self.hypothesis,
                "evidence": list(self.evidence),
                "patch": self.patch,
                "enable_options": self.enable_options,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True)
class BuildArtifacts:
    name: str
    repo: Path
    binary: Path
    image: Path
    nemu: Path
    generated_fingerprint: str
    build_log: Path | None = None
    build_config_fingerprint: str = ""
    toolchain_fingerprint: str = ""
    candidate_proof_id: str = ""
    candidate_proof_sha256: str = ""

    def runtime_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repo": str(self.repo),
            "binary": str(self.binary),
            "image": str(self.image),
            "nemu": str(self.nemu),
            "generated_fingerprint": self.generated_fingerprint,
        }


@dataclass(frozen=True)
class Slot:
    root: Path
    control_repo: Path
    candidate_repo: Path
    results_dir: Path
    lock_file: Any


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise CandidateError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _extract_candidate_payload(text: str) -> str:
    text = text.lstrip("\ufeff")
    starts = list(_MARKER_START_RE.finditer(text))
    ends = list(_MARKER_END_RE.finditer(text))
    if starts or ends:
        if len(starts) != 1 or len(ends) != 1 or starts[0].end() >= ends[0].start():
            raise CandidateError("candidate must contain exactly one ordered EVOLVE-BLOCK")
        text = text[starts[0].end() : ends[0].start()]

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise CandidateError("unterminated candidate code fence")
        stripped = "\n".join(lines[1:-1]).strip()
    return stripped


def parse_candidate_text(text: str) -> Candidate:
    if len(text.encode("utf-8")) > MAX_CANDIDATE_BYTES:
        raise CandidateError("candidate document exceeds size limit")
    payload = _extract_candidate_payload(text)
    try:
        raw = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except CandidateError:
        raise
    except json.JSONDecodeError as exc:
        raise CandidateError(
            f"candidate is not valid JSON at line {exc.lineno}, column {exc.colno}"
        ) from None

    if not isinstance(raw, dict):
        raise CandidateError("candidate must be a JSON object")
    expected = {
        "schema_version",
        "candidate_mode",
        "hypothesis",
        "evidence",
        "patch",
        "enable_options",
    }
    missing = expected - raw.keys()
    extra = raw.keys() - expected
    if missing:
        raise CandidateError(f"candidate missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise CandidateError(f"candidate contains unknown fields: {', '.join(sorted(extra))}")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != SCHEMA_VERSION:
        raise CandidateError(f"schema_version must equal {SCHEMA_VERSION}")

    candidate_mode = raw["candidate_mode"]
    if not isinstance(candidate_mode, str) or candidate_mode not in {
        "control",
        "default-path",
        "explicit-options",
    }:
        raise CandidateError(
            "candidate_mode must be control, default-path, or explicit-options"
        )

    hypothesis = raw["hypothesis"]
    if not isinstance(hypothesis, str) or not hypothesis.strip() or len(hypothesis) > 4000:
        raise CandidateError("hypothesis must be a non-empty string of at most 4000 characters")

    evidence_raw = raw["evidence"]
    if not isinstance(evidence_raw, list) or not all(
        isinstance(item, str) for item in evidence_raw
    ):
        raise CandidateError("evidence must be a non-empty array of strings in schema v2")
    if (
        not 1 <= len(evidence_raw) <= 32
        or any(not item.strip() or len(item) > 4000 for item in evidence_raw)
    ):
        raise CandidateError("evidence must contain 1..32 non-empty items of at most 4000 characters")
    evidence = tuple(item.strip() for item in evidence_raw)

    patch = raw["patch"]
    if not isinstance(patch, str):
        raise CandidateError("patch must be a string")
    validate_patch(patch, allow_empty=True)

    if not isinstance(raw["enable_options"], list):
        raise CandidateError("enable_options must be a name/value array in schema v2")
    enable_options = validate_enable_options(raw["enable_options"])
    mode_shape = (bool(patch), bool(enable_options))
    expected_shape = {
        "control": (False, False),
        "default-path": (True, False),
        "explicit-options": (True, True),
    }[candidate_mode]
    if mode_shape != expected_shape:
        raise CandidateError(
            f"candidate_mode={candidate_mode} requires patch/options presence "
            f"{expected_shape}, got {mode_shape}; only control may omit a patch"
        )
    return Candidate(
        schema_version=SCHEMA_VERSION,
        candidate_mode=candidate_mode,
        hypothesis=hypothesis.strip(),
        evidence=evidence,
        patch=patch,
        enable_options=enable_options,
    )


def parse_candidate_file(program_path: str | os.PathLike[str]) -> Candidate:
    path = Path(program_path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise CandidateError(f"cannot read candidate file: {exc}") from None
    if len(data) > MAX_CANDIDATE_BYTES:
        raise CandidateError("candidate document exceeds size limit")
    try:
        return parse_candidate_text(data.decode("utf-8"))
    except UnicodeDecodeError:
        raise CandidateError("candidate must be UTF-8 text") from None


def validate_enable_options(raw: Any) -> dict[str, bool | int | float | str]:
    if isinstance(raw, list):
        normalized: dict[str, Any] = {}
        for index, item in enumerate(raw):
            if not isinstance(item, dict) or set(item) != {"name", "value"}:
                raise CandidateError(
                    f"enable_options[{index}] must contain exactly name and value"
                )
            name = item["name"]
            if not isinstance(name, str):
                raise CandidateError(f"enable_options[{index}].name must be a string")
            if name in normalized:
                raise CandidateError(f"duplicate enable option: {name!r}")
            normalized[name] = item["value"]
        raw = normalized
    if not isinstance(raw, dict):
        raise CandidateError("enable_options must be an object or name/value array")
    if len(raw) > MAX_ENABLE_OPTIONS:
        raise CandidateError(f"enable_options may contain at most {MAX_ENABLE_OPTIONS} entries")
    out: dict[str, bool | int | float | str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not _OPTION_NAME_RE.fullmatch(key):
            raise CandidateError(f"unsafe enable option name: {key!r}")
        if key not in _ALLOWED_ENABLE_OPTIONS:
            raise CandidateError(
                f"enable option is not in the pinned optimization allowlist: {key!r}"
            )
        if isinstance(value, bool):
            out[key] = value
        elif type(value) is int:
            if abs(value) > 2**53:
                raise CandidateError(f"integer enable option out of range: {key}")
            out[key] = value
        elif type(value) is float:
            if not math.isfinite(value) or abs(value) > 1e15:
                raise CandidateError(f"numeric enable option out of range: {key}")
            out[key] = value
        elif isinstance(value, str) and _OPTION_STRING_RE.fullmatch(value):
            if _MEASUREMENT_VALUE_RE.search(value):
                raise CandidateError(
                    f"enable option {key!r} selects a diagnostic/measurement mode"
                )
            out[key] = value
        else:
            raise CandidateError(
                f"enable option {key!r} must be a finite scalar; paths and shell syntax are forbidden"
            )
    return dict(sorted(out.items()))


def _validate_patch_path(raw: str) -> str:
    if not raw or raw.startswith("/") or raw.startswith("~") or "\\" in raw:
        raise CandidateError(f"unsafe patch path: {raw!r}")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        raise CandidateError(f"control character in patch path: {raw!r}")
    path = PurePosixPath(raw)
    if any(part in {"", ".", "..", ".git"} for part in path.parts):
        raise CandidateError(f"unsafe patch path: {raw!r}")
    normalized = path.as_posix()
    if (
        normalized.startswith(_ALLOWED_PREFIXES)
        and PurePosixPath(normalized).suffix.lower() in _ALLOWED_SOURCE_SUFFIXES
    ):
        return normalized
    raise CandidateError(
        f"patch path outside existing GrhSIM C/C++ source allowlist: {raw!r}"
    )


def _path_from_file_header(line: str) -> str | None:
    raw = line[4:]
    if "\t" in raw:
        raw = raw.split("\t", 1)[0]
    if raw == "/dev/null":
        raise CandidateError("adding or deleting files is forbidden")
    if raw.startswith("a/") or raw.startswith("b/"):
        return _validate_patch_path(raw[2:])
    raise CandidateError(f"non-canonical unified-diff path: {raw!r}")


def validate_patch(patch: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not patch:
        if allow_empty:
            return ()
        raise CandidateError("patch is empty")
    if len(patch.encode("utf-8")) > MAX_PATCH_BYTES:
        raise CandidateError("patch exceeds size limit")
    if "\x00" in patch or "\r" in patch or "\x1b" in patch:
        raise CandidateError("patch contains forbidden binary/control data")
    if not patch.endswith("\n"):
        raise CandidateError("patch must end with a newline")
    if contains_secret(patch):
        raise CandidateError("patch appears to contain an API credential or secret")

    files: list[str] = []
    section_old: str | None = None
    section_new: str | None = None
    saw_old_header = False
    saw_new_header = False
    saw_hunk = False
    for line in patch.splitlines():
        if line.startswith(("diff --cc ", "diff --combined ", "@@@")):
            raise CandidateError("combined diffs are forbidden")
        if line.startswith(_FORBIDDEN_PATCH_LINES):
            raise CandidateError(f"forbidden patch directive: {line.split(maxsplit=2)[0]}")
        if line.startswith(("new file mode ", "deleted file mode ")):
            raise CandidateError("adding or deleting files is forbidden")
        if line.startswith("index "):
            match = _INDEX_RE.fullmatch(line)
            if not match or match.group(1) in {"120000", "160000"}:
                raise CandidateError("invalid or unsafe index/mode line")

        match = _DIFF_HEADER_RE.fullmatch(line)
        if match:
            if section_old is not None and not (saw_old_header and saw_new_header and saw_hunk):
                raise CandidateError("each diff section must contain canonical headers and a hunk")
            old_path = _validate_patch_path(match.group(1))
            new_path = _validate_patch_path(match.group(2))
            if old_path != new_path:
                raise CandidateError("renames and cross-path diffs are forbidden")
            files.append(old_path)
            section_old = old_path
            section_new = new_path
            saw_old_header = saw_new_header = saw_hunk = False
            continue

        if line.startswith("--- ") and not saw_hunk:
            if section_old is None or saw_old_header:
                raise CandidateError("unexpected old-file header")
            header_path = _path_from_file_header(line)
            if header_path is not None and header_path != section_old:
                raise CandidateError("old-file header does not match diff header")
            saw_old_header = True
        elif line.startswith("+++ ") and not saw_hunk:
            if section_new is None or not saw_old_header or saw_new_header:
                raise CandidateError("unexpected new-file header")
            header_path = _path_from_file_header(line)
            if header_path is not None and header_path != section_new:
                raise CandidateError("new-file header does not match diff header")
            saw_new_header = True
        elif line.startswith("@@ "):
            if section_old is None or not (saw_old_header and saw_new_header):
                raise CandidateError("hunk appears before canonical file headers")
            saw_hunk = True

    if section_old is None:
        raise CandidateError("patch has no diff sections")
    if not (saw_old_header and saw_new_header and saw_hunk):
        raise CandidateError("final diff section lacks canonical headers or hunks")
    if len(files) > MAX_PATCH_FILES:
        raise CandidateError(f"patch changes more than {MAX_PATCH_FILES} files")
    if len(files) != len(set(files)):
        raise CandidateError("a file may appear in only one diff section")
    return tuple(files)


def _known_secret_values() -> tuple[str, ...]:
    values: list[str] = []
    for name, value in os.environ.items():
        if _SECRET_ENV_FRAGMENT_RE.search(name) and len(value) >= 8:
            values.append(value)
    return tuple(sorted(set(values), key=len, reverse=True))


def contains_secret(text: str) -> bool:
    if any(secret in text for secret in _known_secret_values()):
        return True
    return any(pattern.search(text) for pattern in _SECRET_TEXT_PATTERNS)


def scrub_secrets(value: Any) -> Any:
    """Recursively redact credential-shaped strings before metrics/log output."""
    if isinstance(value, str):
        out = value
        for secret in _known_secret_values():
            out = out.replace(secret, "[REDACTED]")
        for pattern in _SECRET_TEXT_PATTERNS:
            if pattern.groups:
                out = pattern.sub(lambda match: f"{match.group(1)}[REDACTED]", out)
            else:
                out = pattern.sub("[REDACTED]", out)
        return out
    if isinstance(value, Mapping):
        return {str(key): scrub_secrets(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_secrets(item) for item in value]
    if is_dataclass(value):
        return scrub_secrets(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return scrub_secrets(str(value))


def _clean_subprocess_env(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    env: dict[str, str] = {}
    for name, value in os.environ.items():
        if _SECRET_ENV_FRAGMENT_RE.search(name):
            continue
        if name in _SCRUBBED_ENV_NAMES or name.startswith(_SCRUBBED_ENV_PREFIXES):
            continue
        env[name] = value
    if extra:
        env.update({str(key): str(value) for key, value in extra.items()})
    return env


def _run_sourced(
    env_sh: Path,
    argv: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path,
    timeout: float = DEFAULT_BUILD_TIMEOUT_SECONDS,
    extra_env: Mapping[str, str] | None = None,
    log_path: Path | None = None,
    check: bool = True,
    error_cls: type[InfrastructureError] | type[CandidateError] = InfrastructureError,
) -> subprocess.CompletedProcess[str]:
    """Run an argv without interpolation after sourcing the repository env.sh."""
    if not env_sh.is_file():
        raise InfrastructureError(f"required env.sh is missing: {env_sh}")
    # env.sh intentionally runs pip even when sourced; suppress its setup
    # chatter on both streams so machine-readable command output (notably
    # git rev-parse) cannot be contaminated by pip cache warnings.
    script = 'set -euo pipefail\nsource "$1" >/dev/null 2>&1\nshift\nexec "$@"'
    command = ["bash", "-c", script, "grhsim-evaluator", str(env_sh), *map(str, argv)]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=_clean_subprocess_env(extra_env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise error_cls(f"command timed out after {timeout:.0f}s: {argv[0]}") from exc
    output = scrub_secrets(completed.stdout or "")
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(str(output), encoding="utf-8")
    if check and completed.returncode != 0:
        tail = str(output)[-1600:].replace("\n", " ")
        raise error_cls(
            f"command failed with exit {completed.returncode}: {argv[0]}; output tail: {tail}"
        )
    return subprocess.CompletedProcess(
        completed.args,
        completed.returncode,
        str(output),
        None,
    )


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise InfrastructureError(f"cannot hash non-regular artifact: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_document_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=str,
        )
        + "\n"
    ).encode("utf-8")


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> str:
    """Atomically publish one JSON document and return its byte SHA-256."""

    path.parent.mkdir(parents=True, exist_ok=True)
    data = _json_document_bytes(value)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(
            path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(data).hexdigest()


def _open_regular_lock(path: Path, *, create: bool) -> Any:
    flags = os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
    if create:
        flags |= os.O_CREAT
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise InfrastructureError(f"cannot open evaluator lock safely: {path}: {error}") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise InfrastructureError(f"evaluator lock is not a regular file: {path}")
        return os.fdopen(descriptor, "r+", encoding="utf-8", closefd=True)
    except Exception:
        os.close(descriptor)
        raise


def _generation_input_files(repo: Path) -> tuple[tuple[str, Path], ...]:
    """Return the fixed HDL/difftest inputs consumed by the GrhSIM build."""
    repo_root = repo.resolve()
    rows: list[tuple[str, Path]] = []
    for relative_root, suffixes in _GENERATION_INPUT_ROOTS:
        root = repo_root / relative_root
        if not root.is_dir() or root.is_symlink():
            raise InfrastructureError(f"generation input directory is missing or unsafe: {root}")
        try:
            descendants = sorted(root.rglob("*"))
        except OSError as exc:
            raise InfrastructureError(
                f"failed to enumerate generation inputs under {root}: {exc}"
            ) from None
        for path in descendants:
            if path.is_symlink():
                raise InfrastructureError(f"generation input may not be a symlink: {path}")
            if path.is_dir():
                continue
            if not path.is_file():
                raise InfrastructureError(f"generation input is not a regular file: {path}")
            if suffixes is not None and path.suffix not in suffixes:
                continue
            try:
                relative = path.relative_to(repo_root).as_posix()
            except ValueError:
                raise InfrastructureError(
                    f"generation input escaped its repository: {path}"
                ) from None
            rows.append((relative, path))

    rows.sort(key=lambda row: row[0])
    names = {relative for relative, _path in rows}
    missing = sorted(_REQUIRED_GENERATION_INPUTS - names)
    if missing:
        raise InfrastructureError(f"required generation inputs are missing: {missing}")
    return tuple(rows)


def _generation_input_fingerprint_from_files(files: Sequence[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    digest.update(b"simpletes-grhsim-generation-input-v1\0")
    digest.update(len(files).to_bytes(8, "big"))
    for relative, path in files:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        try:
            size = path.stat().st_size
            digest.update(size.to_bytes(8, "big"))
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise InfrastructureError(f"failed to hash generation input {path}: {exc}") from None
    return digest.hexdigest()


def generation_input_fingerprint(repo: Path) -> str:
    return _generation_input_fingerprint_from_files(_generation_input_files(repo))


def _stage_control_generation_inputs(control_repo: Path, candidate_repo: Path) -> str:
    """Copy a private, byte-exact control RTL snapshot into a fresh candidate."""
    control_files = _generation_input_files(control_repo)
    expected = _generation_input_fingerprint_from_files(control_files)
    candidate_root = candidate_repo.resolve()

    for relative, source in control_files:
        destination = candidate_root / relative
        if destination.exists() or destination.is_symlink():
            raise InfrastructureError(
                f"candidate generation-input destination already exists: {destination}"
            )
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            source_stat = source.stat()
            destination_stat = destination.stat()
        except OSError as exc:
            raise InfrastructureError(
                f"failed to stage generation input {relative}: {exc}"
            ) from None
        if (
            source_stat.st_dev == destination_stat.st_dev
            and source_stat.st_ino == destination_stat.st_ino
        ):
            raise InfrastructureError(
                f"candidate generation input is not a private copy: {relative}"
            )

    control_after = generation_input_fingerprint(control_repo)
    candidate_after = generation_input_fingerprint(candidate_repo)
    if control_after != expected or candidate_after != expected:
        raise InfrastructureError(
            "failed to stage a stable byte-exact control generation-input snapshot"
        )
    return expected


def _verify_generation_inputs_unchanged(
    repo: Path, expected: str, *, phase: str
) -> None:
    actual = generation_input_fingerprint(repo)
    if actual != expected:
        raise InfrastructureError(
            "fixed generation inputs drifted "
            f"{phase}: expected={expected}, actual={actual}"
        )


def _build_config_fingerprint(
    options: Mapping[str, bool | int | float | str], *, jobs: int
) -> str:
    """Fingerprint all code-generation/build settings except log-only RUN_ID."""
    payload = {
        "target": "xs_wolf_grhsim_emu",
        "jobs": jobs,
        "candidate_assignments": [
            f"{name}={value}" for name, value in sorted(_option_environment(options).items())
        ],
        "fixed_assignments": [f"XS_VM_BUILD_JOBS={jobs}", *_BASE_FIXED_MAKE_ASSIGNMENTS],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _toolchain_fingerprint(repo: Path, env_sh: Path) -> str:
    """Hash resolved build-tool executables under the sourced target environment."""
    probe = r'''
import hashlib
import json
from pathlib import Path
import shutil

names = (
    "make", "cmake", "ninja", "python", "python3", "cc", "c++", "gcc", "g++",
    "clang", "clang++", "ld", "ar", "java", "verilator", "mill", "sbt",
)
rows = {}
for name in names:
    found = shutil.which(name)
    if not found:
        rows[name] = None
        continue
    try:
        path = Path(found).resolve(strict=True)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        # Absolute repo-local virtualenv paths differ between the isolated
        # control and candidate clones.  Executable content is the toolchain
        # identity; embedding its workspace path would create a false mismatch.
        rows[name] = {"sha256": digest.hexdigest()}
    except (OSError, RuntimeError) as exc:
        rows[name] = {"error": type(exc).__name__}
print("SIMPLETES_TOOLCHAIN_JSON=" + json.dumps(rows, sort_keys=True, separators=(",", ":")))
'''
    completed = _run_sourced(
        env_sh,
        [sys.executable, "-c", probe],
        cwd=repo,
        timeout=120,
    )
    prefix = "SIMPLETES_TOOLCHAIN_JSON="
    payload = next(
        (line[len(prefix) :] for line in reversed((completed.stdout or "").splitlines()) if line.startswith(prefix)),
        None,
    )
    if payload is None:
        raise InfrastructureError("toolchain fingerprint probe produced no identity")
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise InfrastructureError("toolchain fingerprint probe produced invalid JSON") from exc
    return hashlib.sha256(
        json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _safe_remove_tree(path: Path, allowed_root: Path) -> None:
    root = allowed_root.resolve()
    candidate = path.resolve(strict=False)
    if candidate == root or root not in candidate.parents:
        raise InfrastructureError(f"refusing to remove path outside evaluator slot: {candidate}")
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _git_output(env_sh: Path, repo: Path, args: Sequence[str]) -> str:
    return _run_sourced(env_sh, ["git", "-C", repo, *args], cwd=repo).stdout or ""


def _is_initialized_git_worktree(path: Path, env_sh: Path) -> bool:
    if not path.is_dir():
        return False
    completed = _run_sourced(
        env_sh,
        ["git", "-C", path, "rev-parse", "--show-toplevel"],
        cwd=path,
        check=False,
    )
    if completed.returncode != 0:
        return False
    top = (completed.stdout or "").strip()
    if not top:
        return False
    try:
        return Path(top).resolve() == path.resolve()
    except (OSError, RuntimeError):
        return False


def _clone_repo_recursive(
    source: Path,
    destination: Path,
    commit: str,
    *,
    env_sh: Path,
    root_level: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run_sourced(
        env_sh,
        ["git", "clone", "--shared", "--no-checkout", "--", source, destination],
        cwd=destination.parent,
    )
    _run_sourced(env_sh, ["git", "-C", destination, "checkout", "--detach", commit], cwd=destination)
    head = _git_output(env_sh, destination, ["rev-parse", "HEAD"]).strip()
    if head != commit:
        raise InfrastructureError(f"clone revision mismatch for {destination}: {head}")

    tree = _git_output(env_sh, destination, ["ls-tree", "-r", "-z", commit])
    for entry in tree.split("\0"):
        if not entry:
            continue
        metadata, path = entry.split("\t", 1)
        mode, kind, object_id = metadata.split(" ", 2)
        if mode != "160000" or kind != "commit":
            continue
        if root_level and path not in _ROOT_REQUIRED_SUBMODULES:
            continue
        source_submodule = source / path
        if not _is_initialized_git_worktree(source_submodule, env_sh):
            if root_level:
                raise InfrastructureError(
                    f"required local submodule is not initialized: {source_submodule}"
                )
            # XiangShan intentionally leaves several nested gitlinks
            # uninitialized.  Clone exactly the initialized local graph rather
            # than contacting their network remotes.
            continue
        _clone_repo_recursive(
            source_submodule,
            destination / path,
            object_id,
            env_sh=env_sh,
            root_level=False,
        )


def _copy_env_sh(source_repo: Path, destination_repo: Path) -> Path:
    source = source_repo / "env.sh"
    if not source.is_file() or source.is_symlink():
        raise InfrastructureError(f"source repository has no regular env.sh: {source}")
    destination = destination_repo / "env.sh"
    shutil.copy2(source, destination)
    return destination


def _verify_pinned_repo(repo: Path, env_sh: Path) -> None:
    parent_head = _git_output(env_sh, repo, ["rev-parse", "HEAD"]).strip()
    wolvrix_head = _git_output(env_sh, repo / "wolvrix", ["rev-parse", "HEAD"]).strip()
    if parent_head != PINNED_PARENT_COMMIT or wolvrix_head != PINNED_WOLVRIX_COMMIT:
        raise InfrastructureError(
            f"slot is not pinned (parent={parent_head}, wolvrix={wolvrix_head})"
        )


def _prepare_control_clone(source_repo: Path, slot_root: Path, control_repo: Path) -> Path:
    source_repo = source_repo.resolve()
    source_env = source_repo / "env.sh"
    if (
        not (source_repo / ".git").exists()
        or not source_env.is_file()
        or source_env.is_symlink()
    ):
        raise InfrastructureError(f"invalid GrhSIM source repository: {source_repo}")
    source_env_sha256 = _sha256_file(source_env)
    expected_marker = {
        "schema_version": CONTROL_MARKER_SCHEMA_VERSION,
        "parent_commit": PINNED_PARENT_COMMIT,
        "wolvrix_commit": PINNED_WOLVRIX_COMMIT,
        "source_env_sha256": source_env_sha256,
    }
    ready = control_repo / ".simpletes-control-ready.json"
    if ready.is_file():
        env_sh = control_repo / "env.sh"
        try:
            marker = json.loads(ready.read_text(encoding="utf-8"))
            if marker == expected_marker and _sha256_file(env_sh) == source_env_sha256:
                _verify_pinned_repo(control_repo, env_sh)
                return env_sh
        except Exception:
            pass
        _safe_remove_tree(control_repo, slot_root)
    elif control_repo.exists() or control_repo.is_symlink():
        # A prior process may have been killed between clone and ready-marker
        # publication.  This target is guarded to the evaluator-owned slot.
        _safe_remove_tree(control_repo, slot_root)

    _clone_repo_recursive(
        source_repo,
        control_repo,
        PINNED_PARENT_COMMIT,
        env_sh=source_env,
        root_level=True,
    )
    env_sh = _copy_env_sh(source_repo, control_repo)
    _verify_pinned_repo(control_repo, env_sh)
    ready.write_text(
        json.dumps(expected_marker, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return env_sh


def _prepare_candidate_clone(control_repo: Path, slot_root: Path, candidate_repo: Path) -> Path:
    _safe_remove_tree(candidate_repo, slot_root)
    control_env = control_repo / "env.sh"
    _clone_repo_recursive(
        control_repo,
        candidate_repo,
        PINNED_PARENT_COMMIT,
        env_sh=control_env,
        root_level=True,
    )
    candidate_env = _copy_env_sh(control_repo, candidate_repo)
    _verify_pinned_repo(candidate_repo, candidate_env)
    return candidate_env


@contextmanager
def acquire_slot(source_repo: Path | None = None, slot_root: Path | None = None) -> Iterator[Slot]:
    source_repo = (source_repo or Path(os.environ.get("GRHSIM_SOURCE_REPO", DEFAULT_SOURCE_REPO))).resolve()
    root = (slot_root or Path(os.environ.get("GRHSIM_SLOT_ROOT", DEFAULT_SLOT_ROOT))).resolve()
    namespace = hashlib.sha256(
        f"{source_repo}\0{PINNED_PARENT_COMMIT}\0{PINNED_WOLVRIX_COMMIT}".encode("utf-8")
    ).hexdigest()[:16]
    slot_root_path = root / namespace / "slot-0"
    slot_root_path.mkdir(parents=True, exist_ok=True)
    lock_path = slot_root_path / "lock"
    lock_file = _open_regular_lock(lock_path, create=True)
    timeout = float(os.environ.get("GRHSIM_SLOT_LOCK_TIMEOUT", "43200"))
    started = time.monotonic()
    while True:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() - started >= timeout:
                lock_file.close()
                raise InfrastructureError("timed out waiting for the single GrhSIM evaluator slot")
            time.sleep(0.25)

    try:
        control_repo = slot_root_path / "control"
        candidate_repo = slot_root_path / "candidate"
        results_dir = slot_root_path / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        _prepare_control_clone(source_repo, slot_root_path, control_repo)
        yield Slot(slot_root_path, control_repo, candidate_repo, results_dir, lock_file)
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def _apply_candidate_patch(candidate: Candidate, repo: Path, env_sh: Path) -> Path:
    patch_path = repo.parent / "candidate.patch"
    patch_path.write_text(candidate.patch, encoding="utf-8")
    wolvrix = repo / "wolvrix"
    expected = set(validate_patch(candidate.patch))
    wolvrix_root = wolvrix.resolve()
    for relative in expected:
        source_path = wolvrix / relative
        resolved = source_path.resolve(strict=False)
        if (
            not source_path.is_file()
            or source_path.is_symlink()
            or wolvrix_root not in resolved.parents
        ):
            raise CandidateError(
                f"candidate may modify only existing regular source files: {relative}"
            )
        _run_sourced(
            env_sh,
            ["git", "-C", wolvrix, "cat-file", "-e", f"HEAD:{relative}"],
            cwd=repo,
            error_cls=CandidateError,
        )
    for args in (
        ["git", "-C", wolvrix, "apply", "--check", "--whitespace=error-all", patch_path],
        ["git", "-C", wolvrix, "apply", "--whitespace=error-all", patch_path],
    ):
        _run_sourced(env_sh, args, cwd=repo, error_cls=CandidateError)
    changed: set[str] = set()
    status = _git_output(
        env_sh,
        wolvrix,
        ["status", "--porcelain=v1", "--untracked-files=all", "--"],
    )
    for line in status.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            raise CandidateError("renamed paths are forbidden")
        changed.add(path)
    if changed != expected:
        raise CandidateError(
            f"applied patch changed unexpected paths: expected={sorted(expected)}, actual={sorted(changed)}"
        )
    return patch_path


def _option_environment(options: Mapping[str, bool | int | float | str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in options.items():
        name = f"WOLVRIX_XS_GRHSIM_{key.upper()}"
        if isinstance(value, bool):
            env[name] = "1" if value else "0"
        else:
            env[name] = str(value)
    return env


def _emit_only_fingerprint(
    repo: Path,
    *,
    name: str,
    options: Mapping[str, bool | int | float | str],
    results_dir: Path,
) -> str:
    """Freshly emit an unpatched same-options reference without O3/link."""
    env_sh = repo / "env.sh"
    timeout = float(os.environ.get("GRHSIM_BUILD_TIMEOUT", str(DEFAULT_BUILD_TIMEOUT_SECONDS)))
    jobs = max(1, int(os.environ.get("GRHSIM_BUILD_JOBS", "4")))
    extra_env = _option_environment(options)
    candidate_assignments = [
        f"{env_name}={value}" for env_name, value in sorted(extra_env.items())
    ]
    fixed_assignments = [
        f"XS_VM_BUILD_JOBS={jobs}",
        *_BASE_FIXED_MAKE_ASSIGNMENTS,
        f"RUN_ID=simpletes_{name}",
    ]
    _run_sourced(
        env_sh,
        [
            "make",
            "--no-print-directory",
            f"-j{jobs}",
            "xs_wolf_grhsim_emit",
            *candidate_assignments,
            *fixed_assignments,
        ],
        cwd=repo,
        timeout=timeout,
        extra_env=extra_env,
        log_path=results_dir / f"{name}_emit.log",
        error_cls=CandidateError,
    )
    fingerprint = generated_fingerprint(repo)
    if not fingerprint:
        raise CandidateError("unpatched same-options generated fingerprint is empty")
    return fingerprint


def _artifact_paths(
    repo: Path,
    *,
    error_cls: type[InfrastructureError] | type[CandidateError] = InfrastructureError,
) -> tuple[Path, Path, Path]:
    binary_link = repo / "build" / "xs" / "grhsim" / "grhsim-compile" / "emu"
    image = repo / "testcase" / "xiangshan" / "ready-to-run" / "coremark-2-iteration.bin"
    nemu = repo / "testcase" / "xiangshan" / "ready-to-run" / "riscv64-nemu-interpreter-so"
    for label, path in (("emu", binary_link), ("image", image), ("NEMU", nemu)):
        if not path.exists():
            raise error_cls(f"build did not produce {label}: {path}")
    repo_resolved = repo.resolve()
    binary_was_symlink = binary_link.is_symlink()
    try:
        binary = binary_link.resolve(strict=True)
        resolved_image = image.resolve(strict=True)
        resolved_nemu = nemu.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise error_cls(f"failed to resolve build artifacts: {exc}") from None
    if (
        repo_resolved not in binary.parents
        or not binary.is_file()
        or binary.is_symlink()
    ):
        raise error_cls(f"emu must resolve to a regular ELF inside the isolated repo: {binary}")
    if not binary_was_symlink and not binary_link.is_file():
        raise error_cls(f"emu path is neither a regular file nor an internal link: {binary_link}")
    for label, original, resolved in (
        ("image", image, resolved_image),
        ("NEMU", nemu, resolved_nemu),
    ):
        if (
            original.is_symlink()
            or repo_resolved not in resolved.parents
            or not resolved.is_file()
            or resolved.is_symlink()
        ):
            raise error_cls(
                f"{label} must be a regular file inside the isolated repo: {resolved}"
            )
    return binary, resolved_image, resolved_nemu


def generated_fingerprint(repo: Path) -> str:
    emit_dir = repo / "build" / "xs" / "grhsim" / "grhsim_emit"
    if not emit_dir.is_dir():
        return ""
    paths = sorted(
        path
        for path in emit_dir.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.suffix in {".cpp", ".cc", ".cxx", ".h", ".hpp"}
    )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(emit_dir).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest() if paths else ""


def _run_focused_tests(
    repo: Path,
    env_sh: Path,
    log_path: Path,
    *,
    trusted_tests_repo: Path,
    error_cls: type[InfrastructureError] | type[CandidateError] = InfrastructureError,
) -> None:
    _run_sourced(
        env_sh,
        [
            "python",
            "-m",
            "unittest",
            "discover",
            "-s",
            trusted_tests_repo / "wolvrix" / "tests" / "pybind",
            "-p",
            "test_*.py",
        ],
        cwd=repo,
        log_path=log_path,
        error_cls=error_cls,
    )


def _run_function_gates(
    artifacts: BuildArtifacts,
    results_dir: Path,
    *,
    error_cls: type[InfrastructureError] | type[CandidateError] = InfrastructureError,
) -> None:
    """Run fixed-ASLR 100/10k correctness gates before formal 50k timing."""
    expected = {
        100: ("Core-0 instrCnt = 0, cycleCnt = 96", "Guest cycle spent: 101"),
        10_000: ("Core-0 instrCnt = 458, cycleCnt = 9996", "Guest cycle spent: 10001"),
    }
    allowed_cpus = sorted(os.sched_getaffinity(0))
    if not allowed_cpus:
        raise InfrastructureError("process has no allowed CPU for function gates")
    cpu = allowed_cpus[0]
    env_sh = artifacts.repo / "env.sh"

    personality_output = _run_sourced(
        env_sh,
        ["setarch", "x86_64", "-R", "sh", "-c", "cat /proc/self/personality"],
        cwd=artifacts.repo,
        timeout=30,
        error_cls=error_cls,
    ).stdout
    personalities = re.findall(r"(?m)^([0-9a-fA-F]{8})\s*$", personality_output)
    if personalities != ["00040000"]:
        raise error_cls(
            "ASLR-disable gate failed: expected exactly one personality 00040000"
        )

    for cycles, (counter_line, guest_line) in expected.items():
        log_path = results_dir / f"{artifacts.name}_function_{cycles}.log"
        completed = _run_sourced(
            env_sh,
            [
                "taskset",
                "-c",
                str(cpu),
                "setarch",
                "x86_64",
                "-R",
                artifacts.binary,
                "-i",
                artifacts.image,
                "--diff",
                artifacts.nemu,
                "-b",
                "0",
                "-e",
                "0",
                "-C",
                str(cycles),
            ],
            cwd=artifacts.repo,
            timeout=float(os.environ.get("GRHSIM_FUNCTION_TIMEOUT", "900")),
            extra_env={"EMU_PROGRESS_EVERY_CYCLES": "0"},
            log_path=log_path,
            check=False,
        )
        output = completed.stdout or ""
        negative = re.search(
            r"mismatch|assert|fatal|error|\bfail(?:ed|ure)?\b|bad trap|segmentation|aborted",
            output,
            re.I,
        )
        walltimes = [int(value) for value in _HOST_TIME_RE.findall(output)]
        if (
            completed.returncode != 0
            or counter_line not in output
            or guest_line not in output
            or negative is not None
            or len(walltimes) != 1
            or walltimes[0] <= 0
        ):
            raise error_cls(
                f"{artifacts.name} failed fixed-ASLR {cycles}-cycle function gate; "
                f"see {log_path}"
            )


def _build_variant(
    repo: Path,
    *,
    name: str,
    options: Mapping[str, bool | int | float | str],
    results_dir: Path,
    clean_first: bool = False,
    run_function_gates: bool = True,
    candidate_owned: bool = False,
    trusted_tests_repo: Path | None = None,
) -> BuildArtifacts:
    env_sh = repo / "env.sh"
    timeout = float(os.environ.get("GRHSIM_BUILD_TIMEOUT", str(DEFAULT_BUILD_TIMEOUT_SECONDS)))
    command_error_cls: type[InfrastructureError] | type[CandidateError] = (
        CandidateError if candidate_owned else InfrastructureError
    )
    if clean_first:
        _run_sourced(
            env_sh,
            ["make", "--no-print-directory", "xs_diff_clean"],
            cwd=repo,
            timeout=timeout,
            log_path=results_dir / f"{name}_clean.log",
            error_cls=command_error_cls,
        )

    extra_env = _option_environment(options)
    make_assignments = [
        f"{env_name}={value}"
        for env_name, value in sorted(extra_env.items())
    ]
    jobs = max(1, int(os.environ.get("GRHSIM_BUILD_JOBS", "4")))
    build_config_fingerprint = _build_config_fingerprint(options, jobs=jobs)
    toolchain_fingerprint = _toolchain_fingerprint(repo, env_sh)
    # Candidate-controlled assignments precede all evaluator-owned settings.
    # GNU make therefore cannot use an option to override the fixed build,
    # resume, instrumentation, or measurement configuration.
    fixed_assignments = [
        f"XS_VM_BUILD_JOBS={jobs}",
        *_BASE_FIXED_MAKE_ASSIGNMENTS,
        f"RUN_ID=simpletes_{name}",
    ]
    command: list[str | os.PathLike[str]] = [
        "make",
        "--no-print-directory",
        f"-j{jobs}",
        "xs_wolf_grhsim_emu",
        *make_assignments,
        *fixed_assignments,
    ]
    _run_sourced(
        env_sh,
        command,
        cwd=repo,
        timeout=timeout,
        extra_env=extra_env,
        log_path=results_dir / f"{name}_build.log",
        error_cls=command_error_cls,
    )
    _run_focused_tests(
        repo,
        env_sh,
        results_dir / f"{name}_focused_tests.log",
        trusted_tests_repo=(trusted_tests_repo or repo),
        error_cls=command_error_cls,
    )
    binary, image, nemu = _artifact_paths(repo, error_cls=command_error_cls)
    fingerprint = generated_fingerprint(repo)
    if not fingerprint:
        raise command_error_cls("generated C++/header fingerprint is empty")
    artifacts = BuildArtifacts(
        name=name,
        repo=repo,
        binary=binary,
        image=image,
        nemu=nemu,
        generated_fingerprint=fingerprint,
        build_log=results_dir / f"{name}_build.log",
        build_config_fingerprint=build_config_fingerprint,
        toolchain_fingerprint=toolchain_fingerprint,
    )
    if run_function_gates:
        _run_function_gates(artifacts, results_dir, error_cls=command_error_cls)
    return artifacts


def _control_artifacts(slot: Slot) -> BuildArtifacts:
    marker = slot.results_dir / "control_artifacts.json"
    if marker.is_file() and not marker.is_symlink():
        try:
            artifacts, _artifact_sha256_by_name = _verify_control_artifact_identity(
                slot
            )
            return artifacts
        except Exception:
            pass

    artifacts = _build_variant(
        slot.control_repo,
        name="control",
        options={},
        results_dir=slot.results_dir,
    )
    artifact_sha256 = {
        "binary": _sha256_file(artifacts.binary),
        "image": _sha256_file(artifacts.image),
        "nemu": _sha256_file(artifacts.nemu),
    }
    marker_payload = {
        **scrub_secrets(asdict(artifacts)),
        "schema_version": CONTROL_MARKER_SCHEMA_VERSION,
        "parent_commit": PINNED_PARENT_COMMIT,
        "wolvrix_commit": PINNED_WOLVRIX_COMMIT,
        "artifact_sha256": artifact_sha256,
    }
    _atomic_write_json(marker, marker_payload)
    return artifacts


def _artifact_sha256(artifacts: BuildArtifacts) -> dict[str, str]:
    return {
        "binary": _sha256_file(artifacts.binary),
        "image": _sha256_file(artifacts.image),
        "nemu": _sha256_file(artifacts.nemu),
    }


def _verify_control_artifact_identity(
    slot: Slot, expected: BuildArtifacts | None = None
) -> tuple[BuildArtifacts, dict[str, str]]:
    """Revalidate the immutable control incarnation without rebuilding it.

    Candidate preparation can take hours.  The cached control therefore has to
    be checked again immediately before a candidate proof is published, rather
    than relying only on the check performed before the candidate build.
    """

    marker = slot.results_dir / "control_artifacts.json"
    if marker.is_symlink() or not marker.is_file():
        raise InfrastructureError(f"control artifact marker is not a regular file: {marker}")
    try:
        raw = json.loads(
            marker.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, CandidateError) as error:
        raise InfrastructureError(f"cannot read control artifact marker: {error}") from error
    if not isinstance(raw, dict):
        raise InfrastructureError("control artifact marker is not a JSON object")

    env_sh = slot.control_repo / "env.sh"
    if env_sh.is_symlink() or not env_sh.is_file():
        raise InfrastructureError(f"control env.sh is not a regular file: {env_sh}")
    if env_sh.stat().st_size <= 0:
        raise InfrastructureError(f"control env.sh is empty: {env_sh}")
    _verify_pinned_repo(slot.control_repo, env_sh)
    try:
        artifacts = BuildArtifacts(
            name=str(raw["name"]),
            repo=Path(raw["repo"]),
            binary=Path(raw["binary"]),
            image=Path(raw["image"]),
            nemu=Path(raw["nemu"]),
            generated_fingerprint=str(raw["generated_fingerprint"]),
            build_log=Path(raw["build_log"]) if raw.get("build_log") else None,
            build_config_fingerprint=str(raw["build_config_fingerprint"]),
            toolchain_fingerprint=str(raw["toolchain_fingerprint"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise InfrastructureError(f"malformed control artifact marker: {error}") from error

    binary, image, nemu = _artifact_paths(slot.control_repo)
    current = replace(
        artifacts,
        repo=slot.control_repo,
        binary=binary,
        image=image,
        nemu=nemu,
    )
    jobs = max(1, int(os.environ.get("GRHSIM_BUILD_JOBS", "4")))
    expected_build_config = _build_config_fingerprint({}, jobs=jobs)
    expected_toolchain = _toolchain_fingerprint(slot.control_repo, env_sh)
    artifact_sha256 = _artifact_sha256(current)
    checks = {
        "schema_version": type(raw.get("schema_version")) is int
        and raw.get("schema_version") == CONTROL_MARKER_SCHEMA_VERSION,
        "name": type(raw.get("name")) is str and artifacts.name == "control",
        "repo": type(raw.get("repo")) is str and artifacts.repo == slot.control_repo,
        "parent_commit": type(raw.get("parent_commit")) is str
        and raw.get("parent_commit") == PINNED_PARENT_COMMIT,
        "wolvrix_commit": type(raw.get("wolvrix_commit")) is str
        and raw.get("wolvrix_commit") == PINNED_WOLVRIX_COMMIT,
        "binary_path": artifacts.binary == binary,
        "image_path": artifacts.image == image,
        "nemu_path": artifacts.nemu == nemu,
        "generated_fingerprint": generated_fingerprint(slot.control_repo)
        == artifacts.generated_fingerprint,
        "build_config_fingerprint": artifacts.build_config_fingerprint
        == expected_build_config,
        "toolchain_fingerprint": artifacts.toolchain_fingerprint == expected_toolchain,
        "artifact_sha256": raw.get("artifact_sha256") == artifact_sha256,
    }
    if expected is not None:
        checks.update(
            {
                "expected_name": expected.name == artifacts.name,
                "expected_repo": expected.repo == artifacts.repo,
                "expected_binary_path": expected.binary == artifacts.binary,
                "expected_image_path": expected.image == artifacts.image,
                "expected_nemu_path": expected.nemu == artifacts.nemu,
                "expected_generated_fingerprint": expected.generated_fingerprint
                == artifacts.generated_fingerprint,
                "expected_build_config_fingerprint": expected.build_config_fingerprint
                == artifacts.build_config_fingerprint,
                "expected_toolchain_fingerprint": expected.toolchain_fingerprint
                == artifacts.toolchain_fingerprint,
            }
        )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise InfrastructureError(
            "cached control artifact identity differs: " + ", ".join(failed)
        )
    return current, artifact_sha256


def _canonical_enable_options(options: Mapping[str, bool | int | float | str]) -> str:
    return json.dumps(
        dict(sorted(options.items())),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _candidate_proof_path(slot: Slot, candidate: Candidate) -> Path:
    return slot.results_dir / f"candidate_proof_{candidate.digest[:16]}.json"


def _candidate_artifact_name(candidate: Candidate) -> str:
    suffix = "default_path" if candidate.is_default_path else "explicit_options"
    return f"candidate_{candidate.digest[:12]}_{suffix}"


def _write_candidate_proof(
    candidate: Candidate,
    slot: Slot,
    control: BuildArtifacts,
    enabled: BuildArtifacts,
    *,
    control_artifact_sha256: Mapping[str, str],
    control_env_sha256: str,
) -> BuildArtifacts:
    verified_control, current_control_artifact_sha256 = _verify_control_artifact_identity(
        slot, control
    )
    if current_control_artifact_sha256 != dict(control_artifact_sha256):
        raise InfrastructureError(
            "control artifact SHA-256 identity changed during candidate preparation"
        )
    current_control_env_sha256 = _sha256_file(verified_control.repo / "env.sh")
    if current_control_env_sha256 != control_env_sha256:
        raise InfrastructureError(
            "control env.sh identity changed during candidate preparation"
        )

    candidate_binary, candidate_image, candidate_nemu = _artifact_paths(
        enabled.repo, error_cls=CandidateError
    )
    if (
        candidate_binary != enabled.binary
        or candidate_image != enabled.image
        or candidate_nemu != enabled.nemu
    ):
        raise CandidateError(
            "candidate artifact paths changed after the gated build"
        )
    candidate_artifact_sha256 = _artifact_sha256(enabled)
    for name in ("image", "nemu"):
        if candidate_artifact_sha256[name] != control_artifact_sha256[name]:
            label = "image" if name == "image" else "NEMU"
            raise CandidateError(
                f"candidate and control {label} SHA-256 differ"
            )

    candidate_env = enabled.repo / "env.sh"
    control_env = verified_control.repo / "env.sh"
    candidate_env_sha256 = _sha256_file(candidate_env)
    if candidate_env_sha256 != current_control_env_sha256:
        raise InfrastructureError(
            "candidate and control env.sh differ after artifact preparation"
        )

    proof_id = uuid.uuid4().hex
    payload = {
        "schema_version": CANDIDATE_PROOF_SCHEMA_VERSION,
        "proof_version": CANDIDATE_PROOF_VERSION,
        "proof_id": proof_id,
        "created_time_ns": time.time_ns(),
        "candidate_digest": candidate.digest,
        "candidate_mode": candidate.candidate_mode,
        "patch_sha256": hashlib.sha256(candidate.patch.encode("utf-8")).hexdigest(),
        "canonical_enable_options": _canonical_enable_options(candidate.enable_options),
        "parent_commit": PINNED_PARENT_COMMIT,
        "wolvrix_commit": PINNED_WOLVRIX_COMMIT,
        "control_identity": {
            "generated_fingerprint": control.generated_fingerprint,
            "build_config_fingerprint": control.build_config_fingerprint,
            "toolchain_fingerprint": control.toolchain_fingerprint,
            "artifact_sha256": dict(control_artifact_sha256),
        },
        "repo": str(enabled.repo.resolve()),
        "generated_fingerprint": enabled.generated_fingerprint,
        "build_config_fingerprint": enabled.build_config_fingerprint,
        "toolchain_fingerprint": enabled.toolchain_fingerprint,
        "artifacts": {
            "binary": {
                "path": str(enabled.binary),
                "sha256": candidate_artifact_sha256["binary"],
            },
            "image": {
                "path": str(enabled.image),
                "sha256": candidate_artifact_sha256["image"],
            },
            "nemu": {
                "path": str(enabled.nemu),
                "sha256": candidate_artifact_sha256["nemu"],
            },
        },
        "env_sh": {
            "candidate_path": str(candidate_env),
            "candidate_sha256": candidate_env_sha256,
            "control_path": str(control_env),
            "control_sha256": current_control_env_sha256,
        },
    }
    proof_sha256 = _atomic_write_json(_candidate_proof_path(slot, candidate), payload)
    return replace(
        enabled,
        candidate_proof_id=proof_id,
        candidate_proof_sha256=proof_sha256,
    )


def _prepare_artifacts(candidate: Candidate, slot: Slot) -> tuple[BuildArtifacts, BuildArtifacts]:
    control = _control_artifacts(slot)
    if candidate.is_control:
        return control, BuildArtifacts(
            name="candidate_control",
            repo=control.repo,
            binary=control.binary,
            image=control.image,
            nemu=control.nemu,
            generated_fingerprint=control.generated_fingerprint,
            build_log=control.build_log,
            build_config_fingerprint=control.build_config_fingerprint,
            toolchain_fingerprint=control.toolchain_fingerprint,
        )

    # Keep a pre-candidate-build identity in memory. A candidate-controlled
    # build must not be able to mutate both the control artifacts and their
    # on-disk marker into a coordinated, self-consistent replacement.
    control_artifact_sha256 = _artifact_sha256(control)
    control_env_sha256 = _sha256_file(control.repo / "env.sh")

    candidate_env = _prepare_candidate_clone(slot.control_repo, slot.root, slot.candidate_repo)
    generation_inputs = _stage_control_generation_inputs(
        slot.control_repo, slot.candidate_repo
    )
    option_control_fingerprint: str | None = None
    if candidate.is_explicit_options:
        # Attribute executable changes to the patch rather than to an existing
        # historical knob alone. This gate is meaningful only when the
        # candidate explicitly selects an option.
        option_control_fingerprint = _emit_only_fingerprint(
            slot.candidate_repo,
            name=f"candidate_{candidate.digest[:12]}_unpatched_options",
            options=candidate.enable_options,
            results_dir=slot.results_dir,
        )
        _verify_generation_inputs_unchanged(
            slot.candidate_repo,
            generation_inputs,
            phase="after the unpatched same-options emit",
        )

    _apply_candidate_patch(candidate, slot.candidate_repo, candidate_env)
    if candidate.is_explicit_options:
        disabled = _build_variant(
            slot.candidate_repo,
            name=f"candidate_{candidate.digest[:12]}_disabled",
            options={},
            results_dir=slot.results_dir,
            clean_first=True,
            candidate_owned=True,
            trusted_tests_repo=slot.control_repo,
        )
        _verify_generation_inputs_unchanged(
            slot.candidate_repo,
            generation_inputs,
            phase="after the default-off candidate build",
        )
        if disabled.build_config_fingerprint != control.build_config_fingerprint:
            raise InfrastructureError(
                "default-off candidate and control used different build/link configurations"
            )
        if disabled.toolchain_fingerprint != control.toolchain_fingerprint:
            raise InfrastructureError(
                "default-off candidate and control used different toolchains"
            )
        if disabled.generated_fingerprint != control.generated_fingerprint:
            raise CandidateError(
                "explicit-options candidate changes generated C++ while options are absent; "
                "use candidate_mode=default-path for a native-default change"
            )

    enabled = _build_variant(
        slot.candidate_repo,
        name=_candidate_artifact_name(candidate),
        options=candidate.enable_options,
        results_dir=slot.results_dir,
        clean_first=True,
        candidate_owned=True,
        trusted_tests_repo=slot.control_repo,
    )
    _verify_generation_inputs_unchanged(
        slot.candidate_repo,
        generation_inputs,
        phase="after the enabled candidate build",
    )
    if enabled.generated_fingerprint == control.generated_fingerprint:
        detail = (
            "native default-path patch has no generated executable effect"
            if candidate.is_default_path
            else "explicit options are unsupported or the patch has no executable effect"
        )
        raise CandidateError(f"candidate generated the same C++/headers as control; {detail}")
    if (
        candidate.is_explicit_options
        and enabled.generated_fingerprint == option_control_fingerprint
    ):
        raise CandidateError(
            "explicit-options candidate is byte-identical to the unpatched same-options build; "
            "the patch has no attributable executable effect"
        )
    if enabled.build_config_fingerprint != _build_config_fingerprint(
        candidate.enable_options,
        jobs=max(1, int(os.environ.get("GRHSIM_BUILD_JOBS", "4"))),
    ):
        raise InfrastructureError("candidate build configuration fingerprint is inconsistent")
    if enabled.toolchain_fingerprint != control.toolchain_fingerprint:
        raise InfrastructureError("candidate and control used different toolchains")
    enabled = _write_candidate_proof(
        candidate,
        slot,
        control,
        enabled,
        control_artifact_sha256=control_artifact_sha256,
        control_env_sha256=control_env_sha256,
    )
    return control, enabled


def _load_runtime_module() -> Any:
    runtime_path = TASK_ROOT / "runtime.py"
    if not runtime_path.is_file():
        raise InfrastructureError(f"runtime module is missing: {runtime_path}")
    module_name = "_simpletes_grhsim_simtop_runtime"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, runtime_path)
    if spec is None or spec.loader is None:
        raise InfrastructureError(f"cannot load runtime module: {runtime_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_config(
    module: Any,
    slot: Slot,
    candidate: Candidate,
    *,
    group_order: str,
) -> Any:
    mapping = {
        "source_repo": str(slot.candidate_repo if not candidate.is_control else slot.control_repo),
        "slot_root": str(slot.root),
        "workspace": str(slot.root),
        "env_sh": str(
            (slot.candidate_repo if not candidate.is_control else slot.control_repo) / "env.sh"
        ),
        "results_dir": str(slot.results_dir / f"runtime_{candidate.digest[:16]}"),
        "candidate_spec": {
            "digest": candidate.digest,
            "candidate_mode": candidate.candidate_mode,
            "enable_options": candidate.enable_options,
        },
        "group_order": group_order,
    }
    config_cls = getattr(module, "RuntimeConfig", None)
    if config_cls is None:
        return mapping
    from_mapping = getattr(config_cls, "from_mapping", None)
    if callable(from_mapping):
        return from_mapping(mapping)
    try:
        return config_cls(**mapping)
    except TypeError:
        return mapping


def _runtime_result_to_dict(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        return dict(result)
    if is_dataclass(result):
        return asdict(result)
    for method_name in ("to_dict", "as_dict"):
        method = getattr(result, method_name, None)
        if callable(method):
            converted = method()
            if isinstance(converted, Mapping):
                return dict(converted)
    raise InfrastructureError(f"runtime returned unsupported result type: {type(result).__name__}")


def _invoke_runtime(
    candidate: Candidate,
    slot: Slot,
    control: BuildArtifacts,
    enabled: BuildArtifacts,
    *,
    group_order: str,
) -> dict[str, Any]:
    module = _load_runtime_module()
    evaluate_candidate = getattr(module, "evaluate_candidate", None)
    if not callable(evaluate_candidate):
        raise InfrastructureError("runtime.py must export evaluate_candidate")
    config = _runtime_config(module, slot, candidate, group_order=group_order)
    result = evaluate_candidate(
        enabled.runtime_mapping(),
        config=config,
        control=control.runtime_mapping(),
    )
    return _runtime_result_to_dict(result)


def _run_runtime_with_retries(
    runtime_callable: Callable[
        ..., Mapping[str, Any]
    ],
    candidate: Candidate,
    slot: Slot,
    control: BuildArtifacts,
    enabled: BuildArtifacts,
    *,
    group_order: str,
) -> dict[str, Any]:
    attempts = max(
        0, int(os.environ.get("GRHSIM_INFRA_RETRIES", DEFAULT_INFRA_RETRIES))
    ) + 1
    result: dict[str, Any] = {}
    retry_history: list[dict[str, Any]] = []
    for attempt in range(1, attempts + 1):
        result = _runtime_result_to_dict(
            runtime_callable(
                candidate,
                slot,
                control,
                enabled,
                group_order=group_order,
            )
        )
        retryable = bool(
            result.get("infrastructure_retry") or result.get("retryable_infra")
        )
        if retryable:
            raw_error = result.get("error")
            error = (
                str(raw_error)
                if raw_error is not None
                else "retryable runtime infrastructure outcome"
            )
            retry_history.append(
                {
                    "attempt": attempt,
                    "error": scrub_secrets(error),
                    "diagnostics": scrub_secrets(result.get("diagnostics", {})),
                }
            )
        if not retryable or attempt == attempts:
            break
    if retry_history:
        final_diagnostics = result.get("diagnostics")
        merged_diagnostics = (
            dict(final_diagnostics) if isinstance(final_diagnostics, Mapping) else {}
        )
        merged_diagnostics["runtime_retry_attempts"] = retry_history
        result["diagnostics"] = merged_diagnostics
        if bool(result.get("infrastructure_retry") or result.get("retryable_infra")):
            result["retry_diagnostic"] = "; ".join(
                f"runtime attempt {item['attempt']}/{attempts}: {item['error']}"
                for item in retry_history
            )
    return result


def _sample_walltimes(result: Mapping[str, Any], role: str) -> list[float]:
    values: list[float] = []
    samples = result.get("samples")
    if isinstance(samples, Sequence) and not isinstance(samples, (str, bytes)):
        for sample in samples:
            if not isinstance(sample, Mapping) or sample.get("role") != role:
                continue
            value = sample.get("walltime_ms")
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and number > 0:
                values.append(number)
    elif isinstance(samples, Mapping):
        for nested in samples.values():
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                values.extend(_sample_walltimes({"samples": nested}, role))
    return values


def _combine_promotion_results(
    screen: Mapping[str, Any], promotion: Mapping[str, Any]
) -> dict[str, Any]:
    control_values = _sample_walltimes(screen, "control") + _sample_walltimes(
        promotion, "control"
    )
    candidate_values = _sample_walltimes(screen, "candidate") + _sample_walltimes(
        promotion, "candidate"
    )
    if not control_values:
        control_values = [_walltime_from_result(screen, "control"), _walltime_from_result(promotion, "control")]
    if not candidate_values:
        candidate_values = [
            _walltime_from_result(screen, "candidate"),
            _walltime_from_result(promotion, "candidate"),
        ]
    screen_positive = (
        _walltime_from_result(screen, "candidate")
        <= _walltime_from_result(screen, "control")
    )
    promotion_positive = (
        _walltime_from_result(promotion, "candidate")
        <= _walltime_from_result(promotion, "control")
    )
    return {
        "valid": True,
        "infrastructure_retry": False,
        "retryable_infra": False,
        "control_walltime_ms": float(statistics.fmean(control_values)),
        "candidate_walltime_ms": float(statistics.fmean(candidate_values)),
        "samples": {
            "abba": screen.get("samples", []),
            "baab": promotion.get("samples", []),
        },
        "diagnostics": {
            "screen": screen.get("diagnostics", {}),
            "promotion": promotion.get("diagnostics", {}),
            "promotion_ran": True,
            "direction_consistent_positive": screen_positive and promotion_positive,
        },
    }


def _coerce_positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise InfrastructureError(f"runtime result lacks numeric {name}") from None
    if not math.isfinite(number) or number <= 0:
        raise InfrastructureError(f"runtime result has non-positive {name}")
    return number


def _walltime_from_result(result: Mapping[str, Any], prefix: str) -> float:
    sample_values = _sample_walltimes(result, prefix)
    if sample_values:
        return float(statistics.fmean(sample_values))
    for key in (f"{prefix}_walltime_ms", f"{prefix}_mean_ms", f"{prefix}_wall_ms"):
        if key in result and result[key] is not None:
            return _coerce_positive(result[key], key)
    samples = result.get("samples")
    values: list[float] = []
    if isinstance(samples, Mapping):
        raw_values = samples.get(prefix)
        if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes)):
            for item in raw_values:
                if isinstance(item, Mapping):
                    item = item.get("walltime_ms")
                values.append(_coerce_positive(item, f"samples.{prefix}"))
    if values:
        return float(statistics.fmean(values))
    raise InfrastructureError(f"runtime result has no {prefix} walltime")


def score_runtime_result(result: Mapping[str, Any]) -> dict[str, Any]:
    retryable = bool(result.get("infrastructure_retry") or result.get("retryable_infra"))
    valid = bool(result.get("valid", not retryable))
    if retryable:
        error = scrub_secrets(
            result.get("error", "retryable infrastructure failure")
        )
        raw_retry_diagnostic = result.get("retry_diagnostic")
        retry_diagnostic = (
            scrub_secrets(raw_retry_diagnostic)
            if isinstance(raw_retry_diagnostic, str) and raw_retry_diagnostic
            else error
        )
        return {
            "combined_score": 0.0,
            "validity": 0.0,
            "valid_candidate": 0,
            "infrastructure_retry": 1.0,
            "retryable_infra": 1.0,
            "retry_after_s": 30.0,
            "error": error,
            "retry_diagnostic": retry_diagnostic,
            "diagnostics": scrub_secrets(result.get("diagnostics", {})),
        }
    if not valid:
        return {
            "combined_score": 0.0,
            "validity": 0.0,
            "valid_candidate": 0,
            "infrastructure_retry": 0.0,
            "retryable_infra": 0.0,
            "error": scrub_secrets(result.get("error", "functional or runtime gate failed")),
            "samples": scrub_secrets(result.get("samples", {})),
            "diagnostics": scrub_secrets(result.get("diagnostics", {})),
        }

    control_ms = _walltime_from_result(result, "control")
    candidate_ms = _walltime_from_result(result, "candidate")
    improvement_ms = control_ms - candidate_ms
    improvement_pct = improvement_ms / control_ms * 100.0
    score = control_ms / candidate_ms
    diagnostics = result.get("diagnostics", {})
    if (
        isinstance(diagnostics, Mapping)
        and diagnostics.get("promotion_ran")
        and not diagnostics.get("direction_consistent_positive")
    ):
        score = min(score, 1.0)
    control_samples = _sample_walltimes(result, "control")
    candidate_samples = _sample_walltimes(result, "candidate")
    control_spread_ms = (
        max(control_samples) - min(control_samples) if len(control_samples) >= 2 else 0.0
    )
    candidate_spread_ms = (
        max(candidate_samples) - min(candidate_samples)
        if len(candidate_samples) >= 2
        else 0.0
    )
    return {
        "combined_score": float(score),
        "validity": 1.0,
        "valid_candidate": 1,
        "infrastructure_retry": 0.0,
        "retryable_infra": 0.0,
        "control_walltime_ms": float(control_ms),
        "candidate_walltime_ms": float(candidate_ms),
        "walltime_improvement_ms": float(improvement_ms),
        "walltime_improvement_pct": float(improvement_pct),
        "control_walltime_spread_ms": float(control_spread_ms),
        "control_walltime_spread_pct": float(control_spread_ms / control_ms * 100.0),
        "candidate_walltime_spread_ms": float(candidate_spread_ms),
        "samples": scrub_secrets(result.get("samples", {})),
        "diagnostics": scrub_secrets(diagnostics),
    }


def _failure(error: Exception, *, retryable: bool = False, elapsed: float = 0.0) -> dict[str, Any]:
    message = scrub_secrets(f"{type(error).__name__}: {error}")
    return {
        "combined_score": 0.0,
        "validity": 0.0,
        "valid_candidate": 0,
        "infrastructure_retry": 1.0 if retryable else 0.0,
        "retryable_infra": 1.0 if retryable else 0.0,
        **({"retry_after_s": 30.0} if retryable else {}),
        "eval_time": float(elapsed),
        "error": message,
        **({"retry_diagnostic": message} if retryable else {}),
    }


def _publish_evaluation_attempt(
    slot: Slot,
    candidate: Candidate,
    enabled: BuildArtifacts,
    runtime_result: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish attempt data and compatibility files before the immutable commit point."""

    if candidate.is_control:
        proof_id = CONTROL_CANDIDATE_PROOF_ID
        proof_sha256 = CONTROL_CANDIDATE_PROOF_SHA256
    else:
        proof_id = enabled.candidate_proof_id
        proof_sha256 = enabled.candidate_proof_sha256
        if re.fullmatch(r"[0-9a-f]{32}", proof_id) is None:
            raise InfrastructureError(
                "non-control evaluation lacks a valid candidate proof id"
            )
        if re.fullmatch(r"[0-9a-f]{64}", proof_sha256) is None:
            raise InfrastructureError(
                "non-control evaluation lacks a valid candidate proof SHA-256"
            )

    created_time_ns = time.time_ns()
    attempt_id = f"{created_time_ns:020d}-{os.getpid()}-{uuid.uuid4().hex}"
    common = {
        "attempt_id": attempt_id,
        "candidate_digest": candidate.digest[:16],
        "candidate_digest_full": candidate.digest,
        "candidate_mode": candidate.candidate_mode,
        "candidate_proof_id": proof_id,
        "candidate_proof_sha256": proof_sha256,
    }
    runtime_payload = {
        **dict(scrub_secrets(runtime_result)),
        "schema_version": RUNTIME_RESULT_SCHEMA_VERSION,
        **common,
    }
    evaluation_payload = {
        **dict(scrub_secrets(metrics)),
        "schema_version": EVALUATION_RESULT_SCHEMA_VERSION,
        **common,
    }

    attempts_root = slot.results_dir / "attempts" / candidate.digest
    attempts_root.mkdir(parents=True, exist_ok=True)
    if attempts_root.is_symlink() or not attempts_root.is_dir():
        raise InfrastructureError(f"unsafe evaluation-attempt root: {attempts_root}")
    attempt_dir = attempts_root / attempt_id
    attempt_dir.mkdir(mode=0o700, exist_ok=False)
    runtime_path = attempt_dir / "runtime_result.json"
    evaluation_path = attempt_dir / "evaluation.json"
    runtime_sha256 = _atomic_write_json(runtime_path, runtime_payload)
    evaluation_sha256 = _atomic_write_json(evaluation_path, evaluation_payload)
    completion = {
        "schema_version": EVALUATION_ATTEMPT_SCHEMA_VERSION,
        "attempt_id": attempt_id,
        "created_time_ns": created_time_ns,
        "candidate_digest": candidate.digest,
        "candidate_mode": candidate.candidate_mode,
        "candidate_proof_id": proof_id,
        "candidate_proof_sha256": proof_sha256,
        "runtime_result_sha256": runtime_sha256,
        "evaluation_sha256": evaluation_sha256,
    }
    # These two paths are compatibility aliases for existing SimpleTES runs.
    # A retry never trusts them because a process can stop between publications.
    _atomic_write_json(
        slot.results_dir / f"runtime_result_{candidate.digest[:16]}.json",
        runtime_payload,
    )
    _atomic_write_json(
        slot.results_dir / f"evaluation_{candidate.digest[:16]}.json",
        evaluation_payload,
    )
    # This is the strict commit point. An attempt with failed compatibility
    # publication remains incomplete and is never eligible for runtime reuse.
    _atomic_write_json(attempt_dir / "complete.json", completion)
    return evaluation_payload


def evaluate(
    program_path: str,
    *,
    runtime_fn: Callable[..., Mapping[str, Any]] | None = None,
    artifact_preparer: Callable[[Candidate, Slot], tuple[BuildArtifacts, BuildArtifacts]] | None = None,
    slot_acquirer: Callable[..., Any] | None = None,
    source_repo: str | os.PathLike[str] | None = None,
    slot_root: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Evaluate one structured candidate and return prompt-friendly metrics."""
    started = time.monotonic()
    try:
        candidate = parse_candidate_file(program_path)
    except CandidateError as exc:
        return _failure(exc, elapsed=time.monotonic() - started)

    runtime_callable = runtime_fn or _invoke_runtime
    prepare = artifact_preparer or _prepare_artifacts
    acquire = slot_acquirer or acquire_slot
    try:
        with acquire(
            Path(source_repo) if source_repo is not None else None,
            Path(slot_root) if slot_root is not None else None,
        ) as slot:
            control, enabled = prepare(candidate, slot)
            result = _run_runtime_with_retries(
                runtime_callable,
                candidate,
                slot,
                control,
                enabled,
                group_order="ABBA",
            )
            if (
                not candidate.is_control
                and bool(result.get("valid"))
                and not bool(
                    result.get("infrastructure_retry") or result.get("retryable_infra")
                )
                and _walltime_from_result(result, "candidate")
                < _walltime_from_result(result, "control")
            ):
                reversed_result = _run_runtime_with_retries(
                    runtime_callable,
                    candidate,
                    slot,
                    control,
                    enabled,
                    group_order="BAAB",
                )
                if bool(reversed_result.get("valid")) and not bool(
                    reversed_result.get("infrastructure_retry")
                    or reversed_result.get("retryable_infra")
                ):
                    result = _combine_promotion_results(
                        result, reversed_result
                    )
                else:
                    result = reversed_result
            metrics = score_runtime_result(result)
            metrics["eval_time"] = float(time.monotonic() - started)
            metrics["candidate_digest"] = candidate.digest[:16]
            metrics["candidate_mode"] = candidate.candidate_mode
            metrics["candidate_files"] = len(validate_patch(candidate.patch, allow_empty=True))
            metrics["enable_option_count"] = len(candidate.enable_options)
            metrics["parent_commit"] = PINNED_PARENT_COMMIT[:12]
            metrics["wolvrix_commit"] = PINNED_WOLVRIX_COMMIT[:12]
            metrics["control_generated_fingerprint"] = control.generated_fingerprint[:16]
            metrics["candidate_generated_fingerprint"] = enabled.generated_fingerprint[:16]
            metrics = scrub_secrets(metrics)
            try:
                return _publish_evaluation_attempt(
                    slot, candidate, enabled, result, metrics
                )
            except OSError as exc:
                raise InfrastructureError(
                    f"failed to publish immutable evaluation attempt: {exc}"
                ) from exc
    except CandidateError as exc:
        return _failure(exc, elapsed=time.monotonic() - started)
    except InfrastructureError as exc:
        return _failure(exc, retryable=True, elapsed=time.monotonic() - started)
    except Exception as exc:
        return _failure(exc, retryable=False, elapsed=time.monotonic() - started)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program_path", nargs="?", default=str(TASK_ROOT / "init_program.txt"))
    parser.add_argument("--source-repo", default=None)
    parser.add_argument("--slot-root", default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and summarize the candidate without cloning, building, or running it.",
    )
    args = parser.parse_args()
    if args.validate_only:
        try:
            candidate = parse_candidate_file(args.program_path)
            result = {
                "valid": True,
                "candidate_digest": candidate.digest,
                "candidate_mode": candidate.candidate_mode,
                "files": list(validate_patch(candidate.patch, allow_empty=True)),
                "enable_options": candidate.enable_options,
            }
        except Exception as exc:
            result = {"valid": False, "error": scrub_secrets(f"{type(exc).__name__}: {exc}")}
        print(json.dumps(result, sort_keys=True))
        return 0 if result["valid"] else 2
    print(json.dumps(evaluate(args.program_path, source_repo=args.source_repo, slot_root=args.slot_root)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
