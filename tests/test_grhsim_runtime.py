from __future__ import annotations

from pathlib import Path

import pytest

from datasets.grhsim.simtop_50k.runtime import (
    ArtifactSet,
    CcdTopology,
    CpuTopology,
    GateAssessment,
    GrhSimRuntime,
    InvalidCandidateError,
    Placement,
    RetryableInfrastructureError,
    RuntimeConfig,
    assess_ccd_gate,
    assess_continuous_monitor,
    audit_function_log,
    audit_numa_maps,
    audit_perf_stat,
    audit_process_state,
    build_aslr_probe_command,
    build_first_touch_command,
    build_monitor_command,
    build_mpstat_command,
    build_source_env_command,
    build_workload_command,
    discover_cpu_topology,
    enumerate_ccds,
    format_cpu_list,
    parse_cpu_list,
    parse_mpstat_idle,
    resolve_artifacts,
    select_helper_cpu,
    select_placement,
)


def _records_for_two_ccds() -> tuple[CpuTopology, ...]:
    records = []
    for ccd_index, base in enumerate((0, 16)):
        cpus = tuple(range(base, base + 16))
        for offset, cpu in enumerate(cpus):
            core_offset = offset % 8
            siblings = (base + core_offset, base + core_offset + 8)
            records.append(
                CpuTopology(
                    cpu=cpu,
                    numa_node=ccd_index,
                    socket_id=ccd_index,
                    core_id=core_offset,
                    siblings=siblings,
                    l3_cpus=cpus,
                )
            )
    return tuple(records)


def _placement() -> Placement:
    records = _records_for_two_ccds()
    ccd = enumerate_ccds(records)[0]
    gate = GateAssessment(True, 16, 100.0, 100.0, 100.0, 100.0)
    return Placement(ccd=ccd, cpu=7, sibling=15, helper_cpu=16, gate=gate)


def _valid_log(walltime: int = 74242) -> str:
    return "\n".join(
        (
            "Core 0: EXCEEDING CYCLE/INSTR LIMIT at pc = 0x80001312",
            "Core-0 instrCnt = 73580, cycleCnt = 49996, IPC = 1.471718",
            "Seed=0 Guest cycle spent: 50001 (snapshot note)",
            f"Host time spent: {walltime}ms",
        )
    )


def _valid_perf() -> str:
    return "\n".join(
        (
            "271786395214,,cycles:u,74215055852,100.00,,",
            "164223412470,,instructions:u,74215055852,100.00,0.60,insn per cycle",
            "1242408056267,,de_no_dispatch_per_slot.no_ops_from_frontend:u,74215055852,100.00,,",
            "161041769140,,cpu/de_no_dispatch_per_slot.no_ops_from_frontend,cmask=0x6/u,74215055852,100.00,,",
            "91055932333,,de_no_dispatch_per_slot.backend_stalls:u,74215055852,100.00,,",
            "74215.06,msec,task-clock,74215055852,100.00,0.999,CPUs utilized",
            "763,,context-switches,74215055852,100.00,10.281,/sec",
            "0,,cpu-migrations,74215055852,100.00,0.000,/sec",
        )
    )


def test_cpu_list_round_trip_and_rejects_invalid_range():
    assert parse_cpu_list("0-3,8,10-11") == (0, 1, 2, 3, 8, 10, 11)
    assert format_cpu_list((11, 2, 1, 0, 10, 8, 3)) == "0-3,8,10-11"
    with pytest.raises(ValueError):
        parse_cpu_list("7-3")


def test_discover_ccds_from_l3_and_thread_siblings_sysfs(tmp_path: Path):
    sysfs = tmp_path / "cpu"
    for record in _records_for_two_ccds():
        cpu_dir = sysfs / f"cpu{record.cpu}"
        (cpu_dir / "topology").mkdir(parents=True)
        (cpu_dir / "cache" / "index3").mkdir(parents=True)
        (cpu_dir / f"node{record.numa_node}").mkdir()
        (cpu_dir / "online").write_text("1\n", encoding="utf-8")
        (cpu_dir / "topology" / "physical_package_id").write_text(
            f"{record.socket_id}\n", encoding="utf-8"
        )
        (cpu_dir / "topology" / "core_id").write_text(
            f"{record.core_id}\n", encoding="utf-8"
        )
        (cpu_dir / "topology" / "thread_siblings_list").write_text(
            format_cpu_list(record.siblings), encoding="utf-8"
        )
        (cpu_dir / "cache" / "index3" / "level").write_text("3\n", encoding="utf-8")
        (cpu_dir / "cache" / "index3" / "shared_cpu_list").write_text(
            format_cpu_list(record.l3_cpus), encoding="utf-8"
        )

    records = discover_cpu_topology(sysfs)
    ccds = enumerate_ccds(records)

    assert len(records) == 32
    assert [ccd.cpus for ccd in ccds] == [tuple(range(16)), tuple(range(16, 32))]
    assert ccds[0].physical_cores[0] == (0, 8)
    assert ccds[1].numa_node == 1


def test_enumerate_ccds_rejects_incomplete_l3_group():
    records = _records_for_two_ccds()[:-1]
    ccds = enumerate_ccds(records)
    assert len(ccds) == 1
    assert ccds[0].cpus == tuple(range(16))


def test_parse_mpstat_and_whole_ccd_gate_selects_idlest_physical_core():
    output = "header\n" + "\n".join(
        f"Average: {cpu} 0 0 0 0 0 0 0 0 0 {100 if cpu in (7, 15) else 99}"
        for cpu in range(16)
    )
    idle = parse_mpstat_idle(output)
    ccd = enumerate_ccds(_records_for_two_ccds())[0]
    assessment = assess_ccd_gate(ccd, idle)

    assert assessment.passed
    assert assessment.count == 16
    assert assessment.target_idle == 100.0
    assert assessment.sibling_idle == 100.0


def test_whole_ccd_gate_requires_all_16_and_all_thresholds():
    ccd = enumerate_ccds(_records_for_two_ccds())[0]
    missing = {cpu: 100.0 for cpu in range(15)}
    assert not assess_ccd_gate(ccd, missing).passed

    low_min = {cpu: 100.0 for cpu in range(16)}
    low_min[1] = 94.9
    assert not assess_ccd_gate(ccd, low_min, target=0, sibling=8).passed

    low_pair = {cpu: 99.0 for cpu in range(16)}
    low_pair[8] = 97.9
    assert not assess_ccd_gate(ccd, low_pair, target=0, sibling=8).passed


def test_dynamic_selection_ranks_physical_core_and_helper_is_outside_ccd():
    records = _records_for_two_ccds()
    ccd0, ccd1 = enumerate_ccds(records)
    idle0 = {cpu: 99.0 for cpu in ccd0.cpus}
    idle1 = {cpu: 99.0 for cpu in ccd1.cpus}
    idle0[7] = idle0[15] = 99.8
    idle1[23] = idle1[31] = 100.0

    placement = select_placement(
        (ccd0, ccd1), {ccd0.key: idle0, ccd1.key: idle1}, records
    )

    assert (placement.cpu, placement.sibling) == (23, 31)
    assert placement.ccd == ccd1
    assert placement.helper_cpu not in ccd1.cpus
    assert placement.helper_cpu == 0  # prefers the other NUMA node
    assert select_helper_cpu(ccd0, records) == 16


def test_no_passing_ccd_is_retryable_infrastructure():
    records = _records_for_two_ccds()
    ccds = enumerate_ccds(records)
    idle = {ccd.key: {cpu: 90.0 for cpu in ccd.cpus} for ccd in ccds}
    with pytest.raises(RetryableInfrastructureError):
        select_placement(ccds, idle, records)


def test_continuous_monitor_requires_exact_other_15_and_quiet_sibling():
    placement = _placement()
    idle = {cpu: 99.0 for cpu in placement.ccd.cpus if cpu != placement.cpu}
    assert assess_continuous_monitor(placement, idle).passed
    del idle[0]
    assert not assess_continuous_monitor(placement, idle).passed
    idle = {cpu: 100.0 for cpu in placement.ccd.cpus if cpu != placement.cpu}
    idle[placement.sibling] = 97.9
    assert not assess_continuous_monitor(placement, idle).passed


def test_commands_encode_gate_first_touch_fixed_aslr_and_workload_order(tmp_path: Path):
    placement = _placement()
    artifacts = ArtifactSet(
        "candidate", tmp_path / "emu", tmp_path / "image", tmp_path / "nemu.so"
    )
    gate = build_mpstat_command(16, placement.ccd.cpus, 3)
    monitor = build_monitor_command(placement, 630)
    copy = build_first_touch_command("/src/emu", "/dev/shm/run/.emu.tmp", 7, 0)
    workload = build_workload_command(artifacts, placement, "/tmp/perf.csv")

    assert gate == ["taskset", "-c", "16", "mpstat", "-P", "0-15", "1", "3"]
    assert monitor[:6] == ["taskset", "-c", "16", "mpstat", "-P", "0-6,8-15"]
    assert copy == [
        "taskset",
        "-c",
        "7",
        "numactl",
        "--physcpubind=7",
        "--membind=0",
        "cp",
        "--reflink=never",
        "/src/emu",
        "/dev/shm/run/.emu.tmp",
    ]
    assert build_aslr_probe_command() == [
        "setarch",
        "x86_64",
        "-R",
        "sh",
        "-c",
        "cat /proc/self/personality",
    ]
    assert workload[:7] == [
        "taskset",
        "-c",
        "7",
        "numactl",
        "--physcpubind=7",
        "--membind=0",
        "perf",
    ]
    setarch_index = workload.index("setarch")
    assert workload[setarch_index : setarch_index + 4] == [
        "setarch",
        "x86_64",
        "-R",
        str(artifacts.binary),
    ]
    diff_index = workload.index("--diff")
    assert workload[diff_index + 1] == str(artifacts.nemu)
    assert workload.count(str(artifacts.nemu)) == 1
    assert workload[-2:] == ["-C", "50000"]


def test_source_environment_command_quotes_env_path_positionally():
    command = build_source_env_command("/slot with spaces/env.sh")
    assert command[:2] == ["bash", "-c"]
    assert 'source "$1"' in command[2]
    assert command[-1] == "/slot with spaces/env.sh"


def test_process_audit_requires_single_cpu_personality_and_expected_exe(tmp_path: Path):
    emu = tmp_path / "emu"
    emu.write_bytes(b"binary")
    status = "Name:\temu\nCpus_allowed_list:\t23\nMems_allowed_list:\t0-1\n"

    passed, diagnostics = audit_process_state(
        status, "00040000\n", 23, resolved_exe=emu, expected_exe=emu
    )
    assert passed
    assert diagnostics["cpus_allowed_list"] == [23]

    assert not audit_process_state(status, "00000000\n", 23)[0]
    assert not audit_process_state(status.replace("23", "23-24"), "00040000\n", 23)[0]


def test_numa_maps_requires_binary_and_nemu_page_locality():
    binary = Path("/dev/shm/run/emu")
    nemu = Path("/dev/shm/run/nemu.so")
    text = "\n".join(
        (
            f"00400000 default file={binary} mapped=20020 N0=20000 N1=20",
            f"7f000000 default file={nemu} mapped=200 N0=190 N1=10",
        )
    )
    passed, diagnostics = audit_numa_maps(
        text, binary=binary, nemu=nemu, target_node=0
    )
    assert passed
    assert diagnostics["binary"]["local_ratio"] == pytest.approx(20000 / 20020)

    remote = text.replace("N0=190 N1=10", "N0=189 N1=11")
    assert not audit_numa_maps(remote, binary=binary, nemu=nemu, target_node=0)[0]


def test_perf_audit_requires_pmu_schedule_task_clock_context_and_zero_migration():
    passed, diagnostics = audit_perf_stat(_valid_perf())
    assert passed
    assert diagnostics["cpu_migrations"]["value"] == 0
    assert diagnostics["task_clock"]["cpus_utilized"] == pytest.approx(0.999)

    assert not audit_perf_stat(_valid_perf().replace("0,,cpu-migrations", "1,,cpu-migrations"))[0]
    assert not audit_perf_stat(_valid_perf().replace("763,,context-switches", "2000,,context-switches"))[0]
    assert not audit_perf_stat(_valid_perf().replace("cycles:u,74215055852,100.00", "cycles:u,74215055852,99.80"))[0]
    assert not audit_perf_stat(
        _valid_perf().replace("task-clock,74215055852,100.00", "task-clock,74215055852,99.80"),
        min_scheduled_percent=99.0,
        min_task_clock_scheduled_percent=99.9,
    )[0]


def test_function_audit_requires_all_canonical_50k_fields_and_unique_walltime():
    passed, walltime, diagnostics = audit_function_log(_valid_log())
    assert passed
    assert walltime == 74242
    assert diagnostics["terminal_pc_ok"]

    assert not audit_function_log(_valid_log().replace("0x80001312", "0x80001310"))[0]
    assert not audit_function_log(_valid_log() + "\nHost time spent: 1ms")[0]
    assert not audit_function_log(_valid_log() + "\nfatal mismatch")[0]


def _write_artifacts(repo: Path, name: str) -> dict[str, object]:
    binary = repo / "build/xs/grhsim/grhsim-compile/emu"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"emu")
    binary.chmod(0o755)
    image = repo / "coremark.bin"
    nemu = repo / "nemu.so"
    image.write_bytes(b"image")
    nemu.write_bytes(b"nemu")
    return {"name": name, "repo": repo, "binary": binary, "image": image, "nemu": nemu}


def test_resolve_artifacts_rejects_binary_symlink_outside_isolated_repo(tmp_path: Path):
    outside = tmp_path / "outside-emu"
    outside.write_bytes(b"emu")
    outside.chmod(0o755)
    repo = tmp_path / "repo"
    binary = repo / "build/xs/grhsim/grhsim-compile/emu"
    binary.parent.mkdir(parents=True)
    binary.symlink_to(outside)
    image = repo / "image"
    nemu = repo / "nemu"
    image.write_bytes(b"image")
    nemu.write_bytes(b"nemu")

    with pytest.raises(InvalidCandidateError, match="outside its isolated repo"):
        resolve_artifacts(
            {"repo": repo, "binary": binary, "image": image, "nemu": nemu},
            RuntimeConfig(),
            role="candidate",
        )


def test_resolve_artifacts_rejects_image_symlink_outside_isolated_repo(tmp_path: Path):
    repo = tmp_path / "repo"
    artifacts = _write_artifacts(repo, "candidate")
    outside = tmp_path / "outside-image"
    outside.write_bytes(b"outside")
    Path(artifacts["image"]).unlink()
    Path(artifacts["image"]).symlink_to(outside)

    with pytest.raises(InvalidCandidateError, match="image resolves outside"):
        resolve_artifacts(artifacts, RuntimeConfig(), role="candidate")


def test_runtime_config_accepts_evaluator_aliases_and_ignores_extra_keys(tmp_path: Path):
    config = RuntimeConfig.from_mapping(
        {
            "source_repo": tmp_path / "repo",
            "workspace": tmp_path / "slot",
            "env_sh": tmp_path / "repo/env.sh",
            "unknown_evaluator_field": "ignored",
        }
    )
    assert config.slot_root == tmp_path / "slot"
    assert config.results_dir == tmp_path / "slot/simtop_50k_results"


def test_external_load_is_returned_as_retryable_infrastructure():
    class LoadedRuntime(GrhSimRuntime):
        def _evaluate_impl(self, candidate, control):
            raise RetryableInfrastructureError(
                "external load prevented gate", diagnostics={"gate": "failed"}
            )

    result = LoadedRuntime().evaluate("candidate")
    assert not result["valid"]
    assert result["infrastructure_retry"]
    assert result["retryable_infra"]
    assert result["diagnostics"]["gate"] == "failed"


def test_formal_default_schedule_is_abba_and_placement_is_fixed(tmp_path: Path):
    candidate = _write_artifacts(tmp_path / "candidate", "candidate")
    control = _write_artifacts(tmp_path / "control", "control")
    placement = _placement()
    seen: list[tuple[str, Placement]] = []

    class FakeRuntime(GrhSimRuntime):
        def _prepare_environment(self):
            return {}

        def _require_tools(self, environment):
            return None

        def _select_fixed_placement(self, environment):
            self._placement = placement
            return placement

        def _run_one(self, artifacts, selected, environment, *, role, index):
            seen.append((role, selected))
            return {"role": role, "index": index, "walltime_ms": 90 if role == "candidate" else 100}

    result = FakeRuntime(RuntimeConfig(samples_per_variant=2)).evaluate(
        candidate, control=control
    )

    assert result["valid"]
    assert [role for role, _ in seen] == ["control", "candidate", "candidate", "control"]
    assert all(selected is placement for _, selected in seen)
    assert result["control_walltime_ms"] == 100
    assert result["candidate_walltime_ms"] == 90
    assert result["relative_improvement"] == pytest.approx(0.1)


def test_formal_promotion_can_use_reversed_baab_order(tmp_path: Path):
    candidate = _write_artifacts(tmp_path / "candidate", "candidate")
    control = _write_artifacts(tmp_path / "control", "control")
    placement = _placement()
    seen: list[str] = []

    class FakeRuntime(GrhSimRuntime):
        def _prepare_environment(self):
            return {}

        def _require_tools(self, environment):
            return None

        def _select_fixed_placement(self, environment):
            self._placement = placement
            return placement

        def _run_one(self, artifacts, selected, environment, *, role, index):
            seen.append(role)
            return {"role": role, "index": index, "walltime_ms": 90 if role == "candidate" else 100}

    result = FakeRuntime(
        RuntimeConfig(samples_per_variant=2, group_order="BAAB")
    ).evaluate(candidate, control=control)

    assert result["valid"]
    assert seen == ["candidate", "control", "control", "candidate"]
