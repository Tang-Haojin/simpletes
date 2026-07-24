from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "datasets" / "grhsim" / "simtop_50k"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load("_test_grhsim_evaluator", TASK_ROOT / "evaluator.py")
launcher = _load("_test_grhsim_launcher", TASK_ROOT / "launcher.py")


VALID_PATCH = """diff --git a/lib/example.cpp b/lib/example.cpp
index 1111111..2222222 100644
--- a/lib/example.cpp
+++ b/lib/example.cpp
@@ -1 +1 @@
-old
+new
"""


def _candidate_text(
    *, patch: str = VALID_PATCH, options: object | None = None, evidence=None
) -> str:
    payload = {
        "schema_version": 1,
        "hypothesis": "Avoid redundant work in the enabled GrhSIM path.",
        "evidence": evidence or ["pdocs/grhsim_opt_thj/TNO0001: absolute count=1"],
        "patch": patch,
        "enable_options": options
        if options is not None
        else {"final_terminal_pushforward_policy": "strict"},
    }
    return (
        "# EVOLVE-BLOCK-START\n"
        + json.dumps(payload)
        + "\n# EVOLVE-BLOCK-END\n"
    )


def test_ablation_materializer_preserves_checkpoint_patch_and_options(
    tmp_path: Path,
):
    source_text = _candidate_text(
        options={"active_mask_gap_pack_policy": "targeted-direct"}
    )
    nodes_path = tmp_path / "nodes.json"
    output_path = tmp_path / "gen20.txt"
    nodes_path.write_text(
        json.dumps([{"id": "node-20", "gen_id": 20, "code": source_text}]),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(TASK_ROOT / "materialize_ablation.py"),
            str(nodes_path),
            "--gen-id",
            "20",
            "--label",
            "no-cold-hint",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    source = evaluator.parse_candidate_text(source_text)
    materialized = evaluator.parse_candidate_file(output_path)
    report = json.loads(completed.stdout)
    assert materialized.patch == source.patch
    assert materialized.enable_options == source.enable_options
    assert materialized.digest == report["digest"]
    assert report["node_id"] == "node-20"


def test_seed_and_schema_are_valid_and_pinned():
    candidate = evaluator.parse_candidate_file(TASK_ROOT / "init_program.txt")
    schema = json.loads((TASK_ROOT / "candidate.schema.json").read_text(encoding="utf-8"))

    assert candidate.is_control
    assert schema["properties"]["schema_version"]["const"] == 1
    option_name_schema = schema["properties"]["enable_options"]["items"]["properties"]["name"]
    assert set(option_name_schema["enum"]) == set(evaluator._ALLOWED_ENABLE_OPTIONS)
    assert evaluator.PINNED_PARENT_COMMIT.startswith("b90d204")
    assert evaluator.PINNED_WOLVRIX_COMMIT.startswith("f17e90e")


def test_candidate_parser_accepts_marked_json_and_canonicalizes_evidence():
    candidate = evaluator.parse_candidate_text(_candidate_text(evidence="one fact"))

    assert candidate.evidence == ("one fact",)
    assert candidate.enable_options == {"final_terminal_pushforward_policy": "strict"}
    assert evaluator.validate_patch(candidate.patch) == ("lib/example.cpp",)


def test_candidate_parser_normalizes_structured_output_option_entries():
    candidate = evaluator.parse_candidate_text(
        _candidate_text(
            options=[
                {"name": "final_terminal_pushforward_policy", "value": "strict"}
            ]
        )
    )

    assert candidate.enable_options == {"final_terminal_pushforward_policy": "strict"}


def test_sourced_command_output_excludes_env_setup_stderr(tmp_path: Path):
    env_sh = tmp_path / "env.sh"
    env_sh.write_text('printf "setup warning\\n" >&2\n', encoding="utf-8")

    completed = evaluator._run_sourced(
        env_sh,
        ["printf", "machine-readable"],
        cwd=tmp_path,
    )

    assert completed.stdout == "machine-readable"


@pytest.mark.parametrize(
    "text, message",
    [
        (
            '{"schema_version":1,"schema_version":1,"hypothesis":"x",'
            '"evidence":["x"],"patch":"","enable_options":{}}',
            "duplicate JSON key",
        ),
        (
            _candidate_text(options={}),
            "require both a non-empty patch and enable_options",
        ),
        (
            _candidate_text(options={"bad-option": True}),
            "unsafe enable option name",
        ),
        (
            _candidate_text(options={"final_topo_policy": "../../escape"}),
            "paths and shell syntax are forbidden",
        ),
        (
            _candidate_text(options={"enable_stats": True}),
            "pinned optimization allowlist",
        ),
        (
            _candidate_text(options={"final_topo_policy": "probe"}),
            "diagnostic/measurement mode",
        ),
    ],
)
def test_candidate_parser_rejects_ambiguous_or_unsafe_documents(text: str, message: str):
    with pytest.raises(evaluator.CandidateError, match=message):
        evaluator.parse_candidate_text(text)


@pytest.mark.parametrize(
    "patch, message",
    [
        (
            VALID_PATCH.replace("a/lib/example.cpp", "a/../runner.py").replace(
                "b/lib/example.cpp", "b/../runner.py"
            ),
            "unsafe patch path|source allowlist",
        ),
        (
            VALID_PATCH.replace("lib/example.cpp", "scripts/runner.py"),
            "source allowlist",
        ),
        (
            VALID_PATCH.replace("lib/example.cpp", "tests/example.cpp"),
            "source allowlist",
        ),
        (
            VALID_PATCH.replace("--- a/lib/example.cpp", "--- /dev/null"),
            "adding or deleting files",
        ),
        (
            VALID_PATCH.replace("index 1111111..2222222 100644", "index 1111111..2222222 120000"),
            "unsafe index/mode",
        ),
        (
            VALID_PATCH.replace("index 1111111..2222222 100644", "GIT binary patch"),
            "forbidden patch directive",
        ),
        (
            VALID_PATCH.replace("+new", "+api_key=sk-abcdefghijklmnop"),
            "credential or secret",
        ),
    ],
)
def test_patch_security_rejects_escape_binary_symlink_and_secrets(patch: str, message: str):
    with pytest.raises(evaluator.CandidateError, match=message):
        evaluator.validate_patch(patch)


def test_patch_parser_does_not_confuse_hunk_content_with_file_headers():
    patch = VALID_PATCH.replace("-old\n+new", "--- source-text\n++++ replacement-text")
    assert evaluator.validate_patch(patch) == ("lib/example.cpp",)


def test_secret_scrubber_redacts_env_and_credential_shapes(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "extremely-secret-value")
    payload = {
        "error": "Bearer abcdefghijklmnop and extremely-secret-value",
        "nested": ["api_key=sk-abcdefghijklmnop"],
    }
    scrubbed = evaluator.scrub_secrets(payload)
    rendered = json.dumps(scrubbed)

    assert "extremely-secret-value" not in rendered
    assert "sk-abcdefghijklmnop" not in rendered
    assert rendered.count("[REDACTED]") >= 2


def test_generated_fingerprint_uses_source_headers_but_ignores_pathful_json(tmp_path: Path):
    repo = tmp_path / "repo"
    emit = repo / "build" / "xs" / "grhsim" / "grhsim_emit"
    emit.mkdir(parents=True)
    (emit / "model.cpp").write_text("int x = 1;\n", encoding="utf-8")
    (emit / "model.h").write_text("extern int x;\n", encoding="utf-8")
    (emit / "stats.json").write_text('{"absolute_path":"/one"}\n', encoding="utf-8")
    first = evaluator.generated_fingerprint(repo)
    (emit / "stats.json").write_text('{"absolute_path":"/two"}\n', encoding="utf-8")
    assert evaluator.generated_fingerprint(repo) == first
    (emit / "model.cpp").write_text("int x = 2;\n", encoding="utf-8")
    assert evaluator.generated_fingerprint(repo) != first


def test_artifact_gate_rejects_emu_symlink_resolving_to_another_workspace(tmp_path: Path):
    repo = tmp_path / "repo"
    binary_dir = repo / "build" / "xs" / "grhsim" / "grhsim-compile"
    ready = repo / "testcase" / "xiangshan" / "ready-to-run"
    binary_dir.mkdir(parents=True)
    ready.mkdir(parents=True)
    outside = tmp_path / "old-workspace-emu"
    outside.write_bytes(b"elf")
    (binary_dir / "emu").symlink_to(outside)
    (ready / "coremark-2-iteration.bin").write_bytes(b"image")
    (ready / "riscv64-nemu-interpreter-so").write_bytes(b"nemu")

    with pytest.raises(evaluator.InfrastructureError, match="inside the isolated repo"):
        evaluator._artifact_paths(repo)


def test_artifact_gate_accepts_internal_build_emu_symlink(tmp_path: Path):
    repo = tmp_path / "repo"
    binary_dir = repo / "build" / "xs" / "grhsim" / "grhsim-compile"
    ready = repo / "testcase" / "xiangshan" / "ready-to-run"
    binary_dir.mkdir(parents=True)
    ready.mkdir(parents=True)
    target = binary_dir / "VSimTop"
    target.write_bytes(b"elf")
    (binary_dir / "emu").symlink_to(target.name)
    image = ready / "coremark-2-iteration.bin"
    nemu = ready / "riscv64-nemu-interpreter-so"
    image.write_bytes(b"image")
    nemu.write_bytes(b"nemu")

    assert evaluator._artifact_paths(repo) == (
        target.resolve(),
        image.resolve(),
        nemu.resolve(),
    )


def test_build_variant_appends_evaluator_fixed_assignments_after_candidate(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "env.sh").write_text("true\n", encoding="utf-8")
    captured: list[list[str]] = []

    def fake_run(_env_sh, argv, **_kwargs):
        captured.append(list(map(str, argv)))
        return subprocess.CompletedProcess(argv, 0, "", None)

    monkeypatch.setattr(evaluator, "_run_sourced", fake_run)
    monkeypatch.setattr(evaluator, "_toolchain_fingerprint", lambda *_args: "toolchain")
    monkeypatch.setattr(evaluator, "_run_focused_tests", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        evaluator,
        "_artifact_paths",
        lambda *_args, **_kwargs: (repo / "emu", repo / "image", repo / "nemu"),
    )
    monkeypatch.setattr(evaluator, "generated_fingerprint", lambda *_args: "generated")
    monkeypatch.setenv("GRHSIM_BUILD_JOBS", "4")

    evaluator._build_variant(
        repo,
        name="candidate",
        options={"final_topo_policy": "strict"},
        results_dir=tmp_path / "results",
        run_function_gates=False,
    )

    make_command = captured[0]
    fixed_tail = [
        "XS_VM_BUILD_JOBS=4",
        *evaluator._BASE_FIXED_MAKE_ASSIGNMENTS,
        "RUN_ID=simpletes_candidate",
    ]
    assert make_command[-len(fixed_tail) :] == fixed_tail
    assert make_command.index("WOLVRIX_XS_GRHSIM_FINAL_TOPO_POLICY=strict") < (
        len(make_command) - len(fixed_tail)
    )


def test_prepare_artifacts_rejects_patch_with_only_preexisting_option_effect(
    monkeypatch, tmp_path: Path
):
    events: list[str] = []
    candidate = evaluator.parse_candidate_text(_candidate_text())
    control_repo = tmp_path / "control"
    candidate_repo = tmp_path / "candidate"
    control_repo.mkdir()
    candidate_repo.mkdir()
    env_sh = candidate_repo / "env.sh"
    env_sh.write_text("true\n", encoding="utf-8")
    results = tmp_path / "results"
    results.mkdir()
    control = evaluator.BuildArtifacts(
        name="control",
        repo=control_repo,
        binary=control_repo / "emu",
        image=control_repo / "image",
        nemu=control_repo / "nemu",
        generated_fingerprint="default",
        build_config_fingerprint="fixed",
        toolchain_fingerprint="toolchain",
    )
    slot = evaluator.Slot(tmp_path, control_repo, candidate_repo, results, None)

    monkeypatch.setattr(evaluator, "_control_artifacts", lambda _slot: control)

    def fake_prepare_clone(*_args, **_kwargs):
        events.append("clone")
        return env_sh

    def fake_stage(*_args, **_kwargs):
        events.append("stage")
        return "fixed-inputs"

    def fake_emit(*_args, **_kwargs):
        events.append("option-control")
        return "option"

    def fake_verify(_repo, expected, *, phase):
        assert expected == "fixed-inputs"
        events.append(phase)

    monkeypatch.setattr(evaluator, "_prepare_candidate_clone", fake_prepare_clone)
    monkeypatch.setattr(evaluator, "_stage_control_generation_inputs", fake_stage)
    monkeypatch.setattr(evaluator, "_verify_generation_inputs_unchanged", fake_verify)
    monkeypatch.setattr(
        evaluator,
        "_apply_candidate_patch",
        lambda *_args: events.append("patch"),
    )
    monkeypatch.setattr(evaluator, "_emit_only_fingerprint", fake_emit)

    def fake_build(_repo, *, name, **_kwargs):
        events.append(name.rsplit("_", 1)[-1])
        fingerprint = "default" if name.endswith("_disabled") else "option"
        return evaluator.BuildArtifacts(
            name=name,
            repo=candidate_repo,
            binary=candidate_repo / "emu",
            image=candidate_repo / "image",
            nemu=candidate_repo / "nemu",
            generated_fingerprint=fingerprint,
            build_config_fingerprint="fixed",
            toolchain_fingerprint="toolchain",
        )

    monkeypatch.setattr(evaluator, "_build_variant", fake_build)

    with pytest.raises(evaluator.CandidateError, match="no attributable executable effect"):
        evaluator._prepare_artifacts(candidate, slot)

    assert events == [
        "clone",
        "stage",
        "option-control",
        "after the unpatched same-options emit",
        "patch",
        "disabled",
        "after the default-off candidate build",
        "enabled",
        "after the enabled candidate build",
    ]


def test_control_cache_rebuilds_when_artifact_sha_changes(monkeypatch, tmp_path: Path):
    repo = tmp_path / "control"
    results = tmp_path / "results"
    repo.mkdir()
    results.mkdir()
    (repo / "env.sh").write_text("true\n", encoding="utf-8")
    binary = repo / "emu"
    image = repo / "image"
    nemu = repo / "nemu"
    binary.write_bytes(b"binary-v1")
    image.write_bytes(b"image")
    nemu.write_bytes(b"nemu")
    monkeypatch.setenv("GRHSIM_BUILD_JOBS", "4")
    monkeypatch.setattr(evaluator, "_toolchain_fingerprint", lambda *_args: "toolchain")
    monkeypatch.setattr(
        evaluator,
        "_artifact_paths",
        lambda *_args, **_kwargs: (binary, image, nemu),
    )
    monkeypatch.setattr(evaluator, "generated_fingerprint", lambda *_args: "generated")
    builds = 0

    def fake_build(*_args, **_kwargs):
        nonlocal builds
        builds += 1
        return evaluator.BuildArtifacts(
            name="control",
            repo=repo,
            binary=binary,
            image=image,
            nemu=nemu,
            generated_fingerprint="generated",
            build_config_fingerprint=evaluator._build_config_fingerprint({}, jobs=4),
            toolchain_fingerprint="toolchain",
        )

    monkeypatch.setattr(evaluator, "_build_variant", fake_build)
    slot = evaluator.Slot(tmp_path, repo, tmp_path / "candidate", results, None)

    evaluator._control_artifacts(slot)
    marker = json.loads((results / "control_artifacts.json").read_text(encoding="utf-8"))
    assert marker["schema_version"] == evaluator.CONTROL_MARKER_SCHEMA_VERSION
    assert set(marker["artifact_sha256"]) == {"binary", "image", "nemu"}
    assert builds == 1

    binary.write_bytes(b"binary-v2")
    evaluator._control_artifacts(slot)
    assert builds == 2


def _write_generation_input_fixture(repo: Path) -> tuple[Path, Path, Path]:
    rtl = repo / "build" / "xs" / "rtl" / "rtl"
    generated = repo / "testcase" / "xiangshan" / "build" / "generated-src"
    rtl.mkdir(parents=True)
    generated.mkdir(parents=True)
    simtop = rtl / "SimTop.sv"
    module = rtl / "nested" / "Module.v"
    module.parent.mkdir()
    simtop.write_text("module SimTop; endmodule\n", encoding="utf-8")
    module.write_text("module Module; endmodule\n", encoding="utf-8")
    (rtl / "SimTop.fir").write_text("large excluded FIR\n", encoding="utf-8")
    (rtl / "filelist.f").write_text("ignored metadata\n", encoding="utf-8")
    macros = generated / "DifftestMacros.svh"
    macros.write_text("`define DIFFTEST 1\n", encoding="utf-8")
    (generated / "difftest_profile.json").write_text(
        '{"profile":"fixed"}\n', encoding="utf-8"
    )
    return simtop, module, macros


def test_stage_control_generation_inputs_copies_private_exact_snapshot(tmp_path: Path):
    control = tmp_path / "control"
    candidate = tmp_path / "candidate"
    control.mkdir()
    candidate.mkdir()
    control_simtop, control_module, control_macros = _write_generation_input_fixture(
        control
    )

    expected = evaluator._stage_control_generation_inputs(control, candidate)

    assert evaluator.generation_input_fingerprint(control) == expected
    assert evaluator.generation_input_fingerprint(candidate) == expected
    assert not (candidate / "build" / "xs" / "rtl" / "rtl" / "SimTop.fir").exists()
    assert not (candidate / "build" / "xs" / "rtl" / "rtl" / "filelist.f").exists()
    for source in (control_simtop, control_module, control_macros):
        destination = candidate / source.relative_to(control)
        assert destination.read_bytes() == source.read_bytes()
        assert not destination.is_symlink()
        assert (destination.stat().st_dev, destination.stat().st_ino) != (
            source.stat().st_dev,
            source.stat().st_ino,
        )

    candidate_simtop = candidate / control_simtop.relative_to(control)
    candidate_simtop.write_text("candidate mutation\n", encoding="utf-8")
    assert control_simtop.read_text(encoding="utf-8") == "module SimTop; endmodule\n"


def test_generation_input_manifest_excludes_fir_but_detects_hdl_drift(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _simtop, module, _macros = _write_generation_input_fixture(repo)
    expected = evaluator.generation_input_fingerprint(repo)

    (repo / "build" / "xs" / "rtl" / "rtl" / "SimTop.fir").write_text(
        "different excluded FIR\n", encoding="utf-8"
    )
    assert evaluator.generation_input_fingerprint(repo) == expected

    module.write_text("module Module; wire drift; endmodule\n", encoding="utf-8")
    with pytest.raises(evaluator.InfrastructureError, match="fixed generation inputs drifted"):
        evaluator._verify_generation_inputs_unchanged(
            repo, expected, phase="after fixture mutation"
        )


def test_generation_input_manifest_rejects_symlinks_and_missing_required_files(
    tmp_path: Path,
):
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir()
    _simtop, _module, _macros = _write_generation_input_fixture(unsafe)
    outside = tmp_path / "outside.svh"
    outside.write_text("outside\n", encoding="utf-8")
    (
        unsafe
        / "testcase"
        / "xiangshan"
        / "build"
        / "generated-src"
        / "linked.svh"
    ).symlink_to(outside)
    with pytest.raises(evaluator.InfrastructureError, match="may not be a symlink"):
        evaluator.generation_input_fingerprint(unsafe)

    incomplete = tmp_path / "incomplete"
    rtl = incomplete / "build" / "xs" / "rtl" / "rtl"
    generated = incomplete / "testcase" / "xiangshan" / "build" / "generated-src"
    rtl.mkdir(parents=True)
    generated.mkdir(parents=True)
    (rtl / "Other.sv").write_text("module Other; endmodule\n", encoding="utf-8")
    (generated / "other.h").write_text("#pragma once\n", encoding="utf-8")
    with pytest.raises(evaluator.InfrastructureError, match="required generation inputs"):
        evaluator.generation_input_fingerprint(incomplete)


def test_recursive_clone_enumerates_nested_gitlinks(monkeypatch, tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "wolvrix").mkdir()
    (source / "testcase" / "xiangshan").mkdir(parents=True)
    destination = tmp_path / "destination"
    env_sh = tmp_path / "env.sh"
    env_sh.write_text("true\n", encoding="utf-8")
    calls: list[list[str]] = []

    def fake_run(_env_sh, argv, **kwargs):
        args = list(map(str, argv))
        calls.append(args)
        if args[:3] == ["git", "clone", "--shared"]:
            Path(args[-1]).mkdir(parents=True, exist_ok=True)
        if args[-2:] == ["rev-parse", "--show-toplevel"]:
            return subprocess.CompletedProcess(
                args, 0, str(Path(args[2]).resolve()) + "\n", None
            )
        return subprocess.CompletedProcess(args, 0, "", None)

    def fake_git_output(_env_sh, repo, args):
        if args[:2] == ["rev-parse", "HEAD"]:
            return "root\n" if Path(repo) == destination else "sub\n"
        if args[:3] == ["ls-tree", "-r", "-z"] and Path(repo) == destination:
            return (
                "160000 commit sub\twolvrix\0"
                "160000 commit sub\ttestcase/xiangshan\0"
            )
        if args[:3] == ["ls-tree", "-r", "-z"]:
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(evaluator, "_run_sourced", fake_run)
    monkeypatch.setattr(evaluator, "_git_output", fake_git_output)
    evaluator._clone_repo_recursive(
        source, destination, "root", env_sh=env_sh, root_level=True
    )

    assert any(call[-2:] == [str(source / "wolvrix"), str(destination / "wolvrix")] for call in calls)
    assert any(
        call[-2:]
        == [str(source / "testcase" / "xiangshan"), str(destination / "testcase" / "xiangshan")]
        for call in calls
    )


def test_uninitialized_submodule_directory_is_not_parent_worktree(monkeypatch, tmp_path: Path):
    parent = tmp_path / "parent"
    child = parent / "uninitialized-submodule"
    child.mkdir(parents=True)
    env_sh = tmp_path / "env.sh"
    env_sh.write_text("true\n", encoding="utf-8")
    monkeypatch.setattr(
        evaluator,
        "_run_sourced",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [], 0, str(parent.resolve()) + "\n", None
        ),
    )

    assert not evaluator._is_initialized_git_worktree(child, env_sh)


def test_score_reports_absolute_and_relative_walltime_and_valid_budget_flag():
    metrics = evaluator.score_runtime_result(
        {
            "valid": True,
            "control_walltime_ms": 75_000,
            "candidate_walltime_ms": 72_000,
            "samples": [],
            "diagnostics": {},
        }
    )

    assert metrics["combined_score"] == pytest.approx(75 / 72)
    assert metrics["control_walltime_ms"] == 75_000
    assert metrics["candidate_walltime_ms"] == 72_000
    assert metrics["walltime_improvement_ms"] == 3_000
    assert metrics["walltime_improvement_pct"] == pytest.approx(4.0)
    assert metrics["valid_candidate"] == 1


def _artifacts(tmp_path: Path, name: str):
    repo = tmp_path / name
    repo.mkdir()
    for filename in ("emu", "image", "nemu"):
        (repo / filename).write_bytes(filename.encode())
    return evaluator.BuildArtifacts(
        name=name,
        repo=repo,
        binary=repo / "emu",
        image=repo / "image",
        nemu=repo / "nemu",
        generated_fingerprint=name * 16,
    )


def test_evaluate_promotes_positive_abba_with_reversed_baab(monkeypatch, tmp_path: Path):
    program = tmp_path / "candidate.txt"
    program.write_text(_candidate_text(), encoding="utf-8")
    control = _artifacts(tmp_path, "control")
    candidate_artifacts = _artifacts(tmp_path, "candidate")
    slot = evaluator.Slot(tmp_path, control.repo, candidate_artifacts.repo, tmp_path, None)

    @contextmanager
    def fake_slot(*_args, **_kwargs):
        yield slot

    monkeypatch.setattr(evaluator, "acquire_slot", fake_slot)
    calls = []

    def fake_runtime(
        _candidate, _slot, runtime_control, runtime_candidate, *, group_order
    ):
        calls.append((runtime_control.name, runtime_candidate.name, group_order))
        if group_order == "ABBA":
            return {
                "valid": True,
                "control_walltime_ms": 100.0,
                "candidate_walltime_ms": 90.0,
                "samples": [
                    {"role": "control", "walltime_ms": 100},
                    {"role": "candidate", "walltime_ms": 90},
                    {"role": "candidate", "walltime_ms": 90},
                    {"role": "control", "walltime_ms": 100},
                ],
                "diagnostics": {"order": "ABBA"},
            }
        return {
            "valid": True,
            "control_walltime_ms": 101.0,
            "candidate_walltime_ms": 91.0,
            "samples": [
                {"role": "candidate", "walltime_ms": 91},
                {"role": "control", "walltime_ms": 101},
                {"role": "control", "walltime_ms": 101},
                {"role": "candidate", "walltime_ms": 91},
            ],
            "diagnostics": {"order": "BAAB"},
        }

    metrics = evaluator.evaluate(
        str(program),
        runtime_fn=fake_runtime,
        artifact_preparer=lambda *_args: (control, candidate_artifacts),
    )

    assert calls == [
        ("control", "candidate", "ABBA"),
        ("control", "candidate", "BAAB"),
    ]
    assert metrics["valid_candidate"] == 1
    assert metrics["control_walltime_ms"] == pytest.approx(100.5)
    assert metrics["candidate_walltime_ms"] == pytest.approx(90.5)
    assert metrics["combined_score"] > 1.0
    assert metrics["diagnostics"]["direction_consistent_positive"]


def test_evaluate_retries_infrastructure_without_counting_candidate(monkeypatch, tmp_path: Path):
    program = tmp_path / "candidate.txt"
    program.write_text(_candidate_text(), encoding="utf-8")
    control = _artifacts(tmp_path, "control")
    candidate_artifacts = _artifacts(tmp_path, "candidate")
    slot = evaluator.Slot(tmp_path, control.repo, candidate_artifacts.repo, tmp_path, None)

    @contextmanager
    def fake_slot(*_args, **_kwargs):
        yield slot

    monkeypatch.setattr(evaluator, "acquire_slot", fake_slot)
    monkeypatch.setenv("GRHSIM_INFRA_RETRIES", "2")
    calls = 0

    def retry(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return {
            "valid": False,
            "infrastructure_retry": True,
            "retryable_infra": True,
            "diagnostics": {"gate": "busy CCD"},
        }

    metrics = evaluator.evaluate(
        str(program),
        runtime_fn=retry,
        artifact_preparer=lambda *_args: (control, candidate_artifacts),
    )

    assert calls == 3
    assert metrics["infrastructure_retry"] == 1
    assert metrics["retryable_infra"] == 1
    assert metrics["valid_candidate"] == 0


def test_candidate_build_failure_is_not_retryable(monkeypatch, tmp_path: Path):
    program = tmp_path / "candidate.txt"
    program.write_text(_candidate_text(), encoding="utf-8")
    slot = evaluator.Slot(tmp_path, tmp_path / "control", tmp_path / "candidate", tmp_path, None)

    @contextmanager
    def fake_slot(*_args, **_kwargs):
        yield slot

    monkeypatch.setattr(evaluator, "acquire_slot", fake_slot)

    def fail(*_args):
        raise evaluator.CandidateError("candidate compile failed")

    metrics = evaluator.evaluate(str(program), artifact_preparer=fail)
    assert metrics["valid_candidate"] == 0
    assert metrics["infrastructure_retry"] == 0
    assert "candidate compile failed" in metrics["error"]


def test_launcher_uses_fixed_model_serial_budget_and_only_secret_file_paths(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("GRHSIM_VERIFY_DEFAULT_OFF", "0")
    monkeypatch.setenv("GRHSIM_RUN_FUNCTION_GATES", "0")
    monkeypatch.setenv("GRHSIM_RUN_FOCUSED_TESTS", "0")
    target = tmp_path / "target"
    (target / ".git").mkdir(parents=True)
    (target / "env.sh").write_text("true\n", encoding="utf-8")
    config = tmp_path / "config.mjy.toml"
    auth = tmp_path / "auth.mjy.json"
    config.write_text("model_provider = 'mjy'\n", encoding="utf-8")
    auth.write_text('{"OPENAI_API_KEY":"do-not-leak"}\n', encoding="utf-8")
    args = SimpleNamespace(
        target_repo=target,
        codex_config=config,
        codex_auth=auth,
        init_program=launcher.DEFAULT_INIT_PROGRAM,
        output_path=tmp_path / "checkpoints",
        slot_root=tmp_path / "slots",
        resume=None,
        max_proposals=16,
        valid_target=8,
        eval_timeout=21_600.0,
        llm_timeout=3_000.0,
        max_tokens=32_768,
    )
    command, environment = launcher.build_command(args)
    rendered = " ".join(command)

    assert "--llm-backend codex_exec" in rendered
    assert "--model gpt-5.6-sol" in rendered
    assert "--reasoning-effort ultra" in rendered
    assert "--max-valid-evaluations 8" in rendered
    assert "--max-generations 16" in rendered
    assert "--eval-concurrency 1" in rendered
    assert "--gen-concurrency 1" in rendered
    assert command[:2] == ["bash", "-c"]
    assert 'source "$1"' in command[2]
    assert str(target / "env.sh") in command
    assert environment["GRHSIM_VERIFY_DEFAULT_OFF"] == "1"
    assert environment["GRHSIM_RUN_FUNCTION_GATES"] == "1"
    assert environment["GRHSIM_RUN_FOCUSED_TESTS"] == "1"
    assert "do-not-leak" not in rendered
    assert "do-not-leak" not in json.dumps(environment)


def test_launcher_uses_explicit_best_program_as_initial_seed(tmp_path: Path):
    target = tmp_path / "target"
    (target / ".git").mkdir(parents=True)
    (target / "env.sh").write_text("true\n", encoding="utf-8")
    config = tmp_path / "config.mjy.toml"
    auth = tmp_path / "auth.mjy.json"
    config.write_text("model_provider = 'mjy'\n", encoding="utf-8")
    auth.write_text('{}\n', encoding="utf-8")
    best_program = tmp_path / "previous" / "db_state_161238" / "best_program.txt"
    best_program.parent.mkdir(parents=True)
    best_program.write_text(_candidate_text(), encoding="utf-8")
    args = SimpleNamespace(
        target_repo=target,
        codex_config=config,
        codex_auth=auth,
        init_program=best_program,
        output_path=tmp_path / "checkpoints",
        slot_root=tmp_path / "slots",
        resume=None,
        max_proposals=16,
        valid_target=8,
        eval_timeout=21_600.0,
        llm_timeout=3_000.0,
        max_tokens=32_768,
    )

    command, _environment = launcher.build_command(args)
    init_index = command.index("--init-program")

    assert command[init_index + 1] == str(best_program.resolve())
    assert "--resume" not in command


def test_launcher_accepts_long_bounded_continuation_and_rejects_oversize_budget(
    tmp_path: Path,
):
    target = tmp_path / "target"
    (target / ".git").mkdir(parents=True)
    (target / "env.sh").write_text("true\n", encoding="utf-8")
    config = tmp_path / "config.mjy.toml"
    auth = tmp_path / "auth.mjy.json"
    config.write_text("model_provider = 'mjy'\n", encoding="utf-8")
    auth.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(
        target_repo=target,
        codex_config=config,
        codex_auth=auth,
        init_program=launcher.DEFAULT_INIT_PROGRAM,
        output_path=tmp_path / "checkpoints",
        slot_root=tmp_path / "slots",
        resume=None,
        max_proposals=32,
        valid_target=16,
        eval_timeout=21_600.0,
        llm_timeout=5_400.0,
        max_tokens=32_768,
    )

    command, _environment = launcher.build_command(args)
    rendered = " ".join(command)
    assert "--max-generations 32" in rendered
    assert "--max-valid-evaluations 16" in rendered
    assert "--timeout 5400.0" in rendered

    args.max_proposals = launcher.MAX_PROPOSALS + 1
    with pytest.raises(SystemExit, match=r"--max-proposals must be in 1\.\.64"):
        launcher.build_command(args)


def test_launcher_rejects_symlinked_explicit_initial_seed(tmp_path: Path):
    target = tmp_path / "target"
    (target / ".git").mkdir(parents=True)
    (target / "env.sh").write_text("true\n", encoding="utf-8")
    config = tmp_path / "config.mjy.toml"
    auth = tmp_path / "auth.mjy.json"
    config.write_text("model_provider = 'mjy'\n", encoding="utf-8")
    auth.write_text('{}\n', encoding="utf-8")
    real_seed = tmp_path / "best_program.txt"
    real_seed.write_text(_candidate_text(), encoding="utf-8")
    linked_seed = tmp_path / "linked_best_program.txt"
    linked_seed.symlink_to(real_seed)
    args = SimpleNamespace(
        target_repo=target,
        codex_config=config,
        codex_auth=auth,
        init_program=linked_seed,
        output_path=tmp_path / "checkpoints",
        slot_root=tmp_path / "slots",
        resume=None,
        max_proposals=16,
        valid_target=8,
        eval_timeout=21_600.0,
        llm_timeout=3_000.0,
        max_tokens=32_768,
    )

    with pytest.raises(SystemExit, match="initial program must not be a symbolic link"):
        launcher.build_command(args)


def test_launcher_cli_defaults_to_dataset_initial_program(monkeypatch, capsys):
    captured = {}

    def fake_build_command(args):
        captured["init_program"] = args.init_program
        return ["true"], {}

    monkeypatch.setattr(launcher, "build_command", fake_build_command)
    monkeypatch.setattr(sys, "argv", ["launcher.py", "--dry-run"])

    assert launcher.main() == 0
    assert captured["init_program"] == launcher.DEFAULT_INIT_PROGRAM
    assert capsys.readouterr().out.strip() == "true"
