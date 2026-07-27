from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "datasets" / "grhsim" / "simtop_50k"
OLD_PRE_RWA_PARENT_COMMIT = "fbe4e1cbbfcf45b52960545377020cb761c3ab25"
OLD_PRE_RWA_WOLVRIX_COMMIT = "8f6ba14397b0c3d00cb909153af1c6464f4f1ed9"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load("_test_grhsim_evaluator", TASK_ROOT / "evaluator.py")
launcher = _load("_test_grhsim_launcher", TASK_ROOT / "launcher.py")
retry_runtime = _load("_test_grhsim_retry_runtime", TASK_ROOT / "retry_runtime.py")
sys.modules.setdefault("evaluator", evaluator)
materialize_ablation = _load(
    "_test_grhsim_materialize_ablation", TASK_ROOT / "materialize_ablation.py"
)


VALID_PATCH = """diff --git a/lib/example.cpp b/lib/example.cpp
index 1111111..2222222 100644
--- a/lib/example.cpp
+++ b/lib/example.cpp
@@ -1 +1 @@
-old
+new
"""


def _candidate_text(
    *,
    patch: str = VALID_PATCH,
    options: object | None = None,
    evidence=None,
    candidate_mode: str = "explicit-options",
) -> str:
    option_value = (
        options
        if options is not None
        else {"final_terminal_pushforward_policy": "strict"}
    )
    if isinstance(option_value, dict):
        option_value = [
            {"name": name, "value": value} for name, value in option_value.items()
        ]
    payload = {
        "schema_version": 2,
        "candidate_mode": candidate_mode,
        "hypothesis": "Avoid redundant work in the enabled GrhSIM path.",
        "evidence": evidence or ["pdocs/grhsim_opt_thj/TNO0001: absolute count=1"],
        "patch": patch,
        "enable_options": option_value,
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
        options={"final_terminal_pushforward_policy": "strict"}
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
    assert materialized.candidate_mode == source.candidate_mode
    assert materialized.patch == source.patch
    assert materialized.enable_options == source.enable_options
    assert materialized.digest == report["digest"]
    assert report["node_id"] == "node-20"


def test_ablation_materializer_mechanically_composes_rwa():
    lines = [f"line {index}\n" for index in range(1, 61)]
    baseline = "".join(lines)

    def changed(*replacements: tuple[int, str]) -> str:
        result = list(lines)
        for line_number, value in replacements:
            result[line_number - 1] = f"{value}\n"
        return "".join(result)

    rw_source = changed((10, "RW"))
    rwf_source = changed((10, "RW"), (30, "F"))
    rwfa_source = changed((10, "RW"), (30, "F"), (50, "A"))

    def node(gen_id: int, node_id: str, source: str):
        return {
            "id": node_id,
            "gen_id": gen_id,
            "code": _candidate_text(
                patch=materialize_ablation._canonical_diff(baseline, source),
                options=[],
                candidate_mode="default-path",
            ),
        }

    nodes = [
        node(16, "rw-node", rw_source),
        node(22, "rwf-node", rwf_source),
        node(40, "rwfa-node", rwfa_source),
    ]
    document, report = materialize_ablation._compose_rwa(
        nodes,
        rw_gen=16,
        rwf_gen=22,
        rwfa_gen=40,
        baseline=baseline,
        baseline_wolvrix_commit="synthetic-test-baseline",
        label="rwa",
    )

    candidate = evaluator.parse_candidate_text(document)
    rwa_source = materialize_ablation._apply_patch(baseline, candidate.patch)
    expected = changed((10, "RW"), (50, "A"))
    assert rwa_source == expected
    assert candidate.candidate_mode == "default-path"
    assert candidate.enable_options == {}
    assert report["mode"] == "compose-rwa"
    assert report["baseline_wolvrix_commit"] == "synthetic-test-baseline"
    assert report["composition"]["rw_plus_a_equals_rwfa_minus_f"] is True
    assert report["composition"]["rwa_plus_f_equals_rwfa"] is True
    assert report["composition"]["rwf_plus_a_equals_rwfa"] is True
    assert report["composition"]["rwa_patch_sha256"] == hashlib.sha256(
        candidate.patch.encode("utf-8")
    ).hexdigest()


def test_ablation_materializer_reads_frozen_historical_rwa_baseline(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command, 0, stdout=b"historical source\n", stderr=b""
        )

    monkeypatch.setattr(materialize_ablation.subprocess, "run", fake_run)

    source = materialize_ablation._historical_rwa_source(Path("/wolvrix"))

    assert source == "historical source\n"
    assert materialize_ablation.HISTORICAL_RWA_BASELINE_WOLVRIX_COMMIT == (
        "8f6ba14397b0c3d00cb909153af1c6464f4f1ed9"
    )
    assert calls == [
        (
            [
                "git",
                "-C",
                "/wolvrix",
                "show",
                "8f6ba14397b0c3d00cb909153af1c6464f4f1ed9:"
                "lib/emit/grhsim_cpp.cpp",
            ],
            {"capture_output": True, "check": False},
        )
    ]


def test_seed_and_schema_are_valid_and_pinned():
    candidate = evaluator.parse_candidate_file(TASK_ROOT / "init_program.txt")
    schema = json.loads((TASK_ROOT / "candidate.schema.json").read_text(encoding="utf-8"))
    seed_payload = evaluator._extract_candidate_payload(
        (TASK_ROOT / "init_program.txt").read_text(encoding="utf-8")
    )

    assert candidate.is_control
    assert candidate.candidate_mode == "control"
    assert candidate.patch == ""
    assert candidate.enable_options == {}
    assert any("Native R lowers" in item for item in candidate.evidence)
    assert any("Native W cold-hints" in item for item in candidate.evidence)
    assert any("Native A nests" in item for item in candidate.evidence)
    assert any("MemoryFill F tier is not present" in item for item in candidate.evidence)
    assert schema["properties"]["schema_version"]["const"] == 2
    assert schema["properties"]["candidate_mode"]["enum"] == [
        "default-path",
        "explicit-options",
    ]
    output_errors = list(
        Draft202012Validator(schema).iter_errors(json.loads(seed_payload))
    )
    assert any(error.validator == "enum" for error in output_errors)
    validated = subprocess.run(
        [
            sys.executable,
            str(TASK_ROOT / "evaluator.py"),
            str(TASK_ROOT / "init_program.txt"),
            "--validate-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["candidate_mode"] == "control"
    option_name_schema = schema["properties"]["enable_options"]["items"]["properties"]["name"]
    assert set(option_name_schema["enum"]) == set(evaluator._ALLOWED_ENABLE_OPTIONS)
    assert evaluator._option_environment(
        {"commit_exact_event_policy": "targeted-cold-layout"}
    ) == {
        "WOLVRIX_XS_GRHSIM_COMMIT_EXACT_EVENT_POLICY": "targeted-cold-layout"
    }
    assert evaluator.PINNED_PARENT_COMMIT == (
        "d31118bea0feb563ad09476e1419f0f15aaf574f"
    )
    assert evaluator.PINNED_WOLVRIX_COMMIT == (
        "16a9f493687a21a5428f1e1327a69834ea60c9f5"
    )
    assert evaluator.PINNED_PARENT_COMMIT != OLD_PRE_RWA_PARENT_COMMIT
    assert evaluator.PINNED_WOLVRIX_COMMIT != OLD_PRE_RWA_WOLVRIX_COMMIT


def test_model_output_schema_rejects_control_proposals():
    schema = json.loads((TASK_ROOT / "candidate.schema.json").read_text(encoding="utf-8"))
    proposal = json.loads(
        evaluator._extract_candidate_payload(
            _candidate_text(
                patch="", options=[], candidate_mode="control", evidence=["control"]
            )
        )
    )

    errors = list(Draft202012Validator(schema).iter_errors(proposal))

    assert errors
    assert any(
        list(error.absolute_path) == ["candidate_mode"] and error.validator == "enum"
        for error in errors
    )


def test_provider_schema_stays_flat_while_local_overlay_rejects_k3_shapes():
    schema = json.loads((TASK_ROOT / "candidate.schema.json").read_text(encoding="utf-8"))
    local_schema = json.loads(
        (TASK_ROOT / "candidate.local.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator.check_schema(local_schema)
    provider_validator = Draft202012Validator(schema)
    local_validator = Draft202012Validator(local_schema)
    valid = json.loads(evaluator._extract_candidate_payload(_candidate_text()))

    assert list(provider_validator.iter_errors(valid)) == []
    assert list(local_validator.iter_errors(valid)) == []

    default_path = dict(valid)
    default_path["candidate_mode"] = "default-path"
    default_path["enable_options"] = []
    assert list(provider_validator.iter_errors(default_path)) == []
    assert list(local_validator.iter_errors(default_path)) == []

    # The production provider has rejected minLength/minItems/pattern and
    # top-level conditionals in prior real requests. Keep its schema flat and
    # apply these complete Draft 2020-12 semantics only after the response is
    # local, where a failure can trigger bounded repair before evaluation.
    invalid_changes = [
        ("empty hypothesis", {"hypothesis": ""}),
        ("whitespace hypothesis", {"hypothesis": " \t"}),
        ("placeholder hypothesis", {"hypothesis": "placeholder"}),
        ("case-insensitive placeholder hypothesis", {"hypothesis": "DUMMY."}),
        ("empty evidence", {"evidence": []}),
        ("empty evidence item", {"evidence": [""]}),
        ("placeholder evidence item", {"evidence": [" Placeholder. "]}),
        ("empty patch", {"patch": ""}),
        ("placeholder patch without newline", {"patch": "placeholder"}),
        ("patch without trailing newline", {"patch": VALID_PATCH.rstrip("\n")}),
        (
            "default-path with options",
            {"candidate_mode": "default-path"},
        ),
        (
            "explicit-options without options",
            {"candidate_mode": "explicit-options", "enable_options": []},
        ),
    ]

    for label, changes in invalid_changes:
        payload = dict(valid)
        payload.update(changes)
        assert list(provider_validator.iter_errors(payload)) == [], label
        assert list(local_validator.iter_errors(payload)), label


def test_model_output_schema_uses_provider_compatible_flat_contract():
    schema = json.loads((TASK_ROOT / "candidate.schema.json").read_text(encoding="utf-8"))
    unsupported = {
        "oneOf",
        "allOf",
        "not",
        "if",
        "then",
        "else",
        "dependentRequired",
        "dependentSchemas",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
    }

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                assert key not in unsupported
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    Draft202012Validator.check_schema(schema)
    visit(schema)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert "anyOf" not in schema


def test_candidate_parser_accepts_marked_json_and_canonicalizes_evidence():
    candidate = evaluator.parse_candidate_text(_candidate_text(evidence=["one fact"]))

    assert candidate.evidence == ("one fact",)
    assert candidate.enable_options == {"final_terminal_pushforward_policy": "strict"}
    assert evaluator.validate_patch(candidate.patch) == ("lib/example.cpp",)

    with pytest.raises(evaluator.CandidateError, match="array of strings"):
        evaluator.parse_candidate_text(_candidate_text(evidence="one fact"))

    for invalid_evidence in (
        ["one fact", " "],
        [f"fact {index}" for index in range(33)],
        [" " * 4001],
    ):
        with pytest.raises(evaluator.CandidateError, match="1..32 non-empty items"):
            evaluator.parse_candidate_text(_candidate_text(evidence=invalid_evidence))


def test_candidate_parser_normalizes_structured_output_option_entries():
    candidate = evaluator.parse_candidate_text(
        _candidate_text(
            options=[
                {"name": "final_terminal_pushforward_policy", "value": "strict"}
            ]
        )
    )

    assert candidate.enable_options == {"final_terminal_pushforward_policy": "strict"}


def test_candidate_parser_accepts_all_v2_modes_and_cross_checks_shapes():
    control = evaluator.parse_candidate_text(
        _candidate_text(
            patch="", options=[], candidate_mode="control", evidence=["control"]
        )
    )
    default_path = evaluator.parse_candidate_text(
        _candidate_text(options=[], candidate_mode="default-path")
    )
    explicit = evaluator.parse_candidate_text(_candidate_text())

    assert control.is_control
    assert default_path.is_default_path and not default_path.enable_options
    assert explicit.is_explicit_options and explicit.enable_options

    for text, mode in (
        (_candidate_text(patch="", options=[], candidate_mode="default-path"), "default-path"),
        (_candidate_text(options=[], candidate_mode="explicit-options"), "explicit-options"),
        (_candidate_text(candidate_mode="control"), "control"),
    ):
        with pytest.raises(evaluator.CandidateError, match=f"candidate_mode={mode}"):
            evaluator.parse_candidate_text(text)

    invalid_mode = json.loads(evaluator._extract_candidate_payload(_candidate_text()))
    invalid_mode["candidate_mode"] = []
    with pytest.raises(evaluator.CandidateError, match="candidate_mode must be"):
        evaluator.parse_candidate_text(json.dumps(invalid_mode))

    object_options = json.loads(evaluator._extract_candidate_payload(_candidate_text()))
    object_options["enable_options"] = {
        "final_terminal_pushforward_policy": "strict"
    }
    with pytest.raises(evaluator.CandidateError, match="name/value array"):
        evaluator.parse_candidate_text(json.dumps(object_options))


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
            '{"schema_version":2,"schema_version":2,"candidate_mode":"control",'
            '"hypothesis":"x","evidence":["x"],"patch":"","enable_options":[]}',
            "duplicate JSON key",
        ),
        (
            _candidate_text(options={}),
            "candidate_mode=explicit-options requires patch/options presence",
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

    captured.clear()
    evaluator._build_variant(
        repo,
        name="native-control",
        options={},
        results_dir=tmp_path / "results",
        run_function_gates=False,
    )
    native_command = captured[0]
    assert not any(
        item.startswith("WOLVRIX_XS_GRHSIM_COMMIT_EXACT_EVENT_POLICY=")
        or item.startswith("WOLVRIX_XS_GRHSIM_ACTIVE_MASK_GAP_PACK_POLICY=")
        for item in native_command
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
    (control_repo / "env.sh").write_text("true\n", encoding="utf-8")
    for filename in ("emu", "image", "nemu"):
        (control_repo / filename).write_bytes(filename.encode())
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
        events.append("disabled" if name.endswith("_disabled") else "candidate-build")
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
        "candidate-build",
        "after the enabled candidate build",
    ]


def test_prepare_artifacts_default_path_uses_native_options_and_skips_option_attribution(
    monkeypatch, tmp_path: Path
):
    events: list[str] = []
    candidate = evaluator.parse_candidate_text(
        _candidate_text(options=[], candidate_mode="default-path")
    )
    control_repo = tmp_path / "control"
    candidate_repo = tmp_path / "candidate"
    control_repo.mkdir()
    candidate_repo.mkdir()
    env_sh = candidate_repo / "env.sh"
    env_sh.write_text("true\n", encoding="utf-8")
    (control_repo / "env.sh").write_text("true\n", encoding="utf-8")
    for filename in ("emu", "image", "nemu"):
        (control_repo / filename).write_bytes(filename.encode())
    results = tmp_path / "results"
    results.mkdir()
    control = evaluator.BuildArtifacts(
        name="control",
        repo=control_repo,
        binary=control_repo / "emu",
        image=control_repo / "image",
        nemu=control_repo / "nemu",
        generated_fingerprint="control-generated",
        build_config_fingerprint="fixed",
        toolchain_fingerprint="toolchain",
    )
    slot = evaluator.Slot(tmp_path, control_repo, candidate_repo, results, None)

    monkeypatch.setattr(evaluator, "_control_artifacts", lambda _slot: control)
    monkeypatch.setattr(
        evaluator,
        "_prepare_candidate_clone",
        lambda *_args, **_kwargs: (events.append("clone") or env_sh),
    )
    monkeypatch.setattr(
        evaluator,
        "_stage_control_generation_inputs",
        lambda *_args, **_kwargs: (events.append("stage") or "fixed-inputs"),
    )
    monkeypatch.setattr(
        evaluator,
        "_verify_generation_inputs_unchanged",
        lambda _repo, expected, *, phase: events.append(phase),
    )
    monkeypatch.setattr(
        evaluator,
        "_apply_candidate_patch",
        lambda *_args: events.append("patch"),
    )
    monkeypatch.setattr(
        evaluator,
        "_emit_only_fingerprint",
        lambda *_args, **_kwargs: pytest.fail("default-path emitted unpatched options"),
    )
    monkeypatch.setattr(
        evaluator,
        "_build_config_fingerprint",
        lambda _options, *, jobs: "fixed",
    )

    def fake_build(_repo, *, name, options, **_kwargs):
        assert name == evaluator._candidate_artifact_name(candidate)
        assert options == {}
        events.append("native-build")
        return evaluator.BuildArtifacts(
            name=name,
            repo=candidate_repo,
            binary=candidate_repo / "emu",
            image=candidate_repo / "image",
            nemu=candidate_repo / "nemu",
            generated_fingerprint="candidate-generated",
            build_config_fingerprint="fixed",
            toolchain_fingerprint="toolchain",
        )

    monkeypatch.setattr(evaluator, "_build_variant", fake_build)
    monkeypatch.setattr(
        evaluator,
        "_write_candidate_proof",
        lambda _candidate, _slot, _control, enabled, **_kwargs: enabled,
    )

    returned_control, returned = evaluator._prepare_artifacts(candidate, slot)

    assert returned_control is control
    assert returned.generated_fingerprint == "candidate-generated"
    assert events == [
        "clone",
        "stage",
        "patch",
        "native-build",
        "after the enabled candidate build",
    ]


def test_prepare_artifacts_default_path_requires_generated_difference(
    monkeypatch, tmp_path: Path
):
    candidate = evaluator.parse_candidate_text(
        _candidate_text(options=[], candidate_mode="default-path")
    )
    control_repo = tmp_path / "control"
    candidate_repo = tmp_path / "candidate"
    control_repo.mkdir()
    candidate_repo.mkdir()
    env_sh = candidate_repo / "env.sh"
    env_sh.write_text("true\n", encoding="utf-8")
    (control_repo / "env.sh").write_text("true\n", encoding="utf-8")
    for filename in ("emu", "image", "nemu"):
        (control_repo / filename).write_bytes(filename.encode())
    results = tmp_path / "results"
    results.mkdir()
    control = evaluator.BuildArtifacts(
        name="control",
        repo=control_repo,
        binary=control_repo / "emu",
        image=control_repo / "image",
        nemu=control_repo / "nemu",
        generated_fingerprint="same",
        build_config_fingerprint="fixed",
        toolchain_fingerprint="toolchain",
    )
    slot = evaluator.Slot(tmp_path, control_repo, candidate_repo, results, None)
    monkeypatch.setattr(evaluator, "_control_artifacts", lambda _slot: control)
    monkeypatch.setattr(evaluator, "_prepare_candidate_clone", lambda *_args: env_sh)
    monkeypatch.setattr(evaluator, "_stage_control_generation_inputs", lambda *_args: "inputs")
    monkeypatch.setattr(evaluator, "_verify_generation_inputs_unchanged", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluator, "_apply_candidate_patch", lambda *_args: None)
    monkeypatch.setattr(
        evaluator,
        "_build_variant",
        lambda *_args, **kwargs: evaluator.BuildArtifacts(
            name=kwargs["name"],
            repo=candidate_repo,
            binary=candidate_repo / "emu",
            image=candidate_repo / "image",
            nemu=candidate_repo / "nemu",
            generated_fingerprint="same",
            build_config_fingerprint="fixed",
            toolchain_fingerprint="toolchain",
        ),
    )

    with pytest.raises(evaluator.CandidateError, match="native default-path patch"):
        evaluator._prepare_artifacts(candidate, slot)


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
        candidate_proof_id="1" * 32,
        candidate_proof_sha256="2" * 64,
    )


def _publish_retry_attempt(proof, *, error="quiet CCD"):
    runtime = {
        "valid": False,
        "infrastructure_retry": True,
        "retryable_infra": True,
        "error": error,
        "samples": [],
    }
    metrics = {
        "combined_score": 0.0,
        "validity": 0.0,
        "valid_candidate": 0,
        "infrastructure_retry": 1.0,
        "retryable_infra": 1.0,
        "retry_after_s": 30.0,
        "error": error,
        "candidate_mode": proof.candidate.candidate_mode,
        "candidate_files": 1,
        "enable_option_count": len(proof.candidate.enable_options),
        "parent_commit": proof.ev.PINNED_PARENT_COMMIT[:12],
        "wolvrix_commit": proof.ev.PINNED_WOLVRIX_COMMIT[:12],
        "control_generated_fingerprint": proof.control.generated_fingerprint[:16],
        "candidate_generated_fingerprint": proof.enabled.generated_fingerprint[:16],
    }
    result = proof.ev._publish_evaluation_attempt(
        proof.slot, proof.candidate, proof.enabled, runtime, metrics
    )
    attempts_root = proof.slot.results_dir / "attempts" / proof.candidate.digest
    attempt_dir = attempts_root / result["attempt_id"]
    proof.attempt_dir = attempt_dir
    proof.runtime_path = attempt_dir / "runtime_result.json"
    proof.evaluation_path = attempt_dir / "evaluation.json"
    proof.complete_path = attempt_dir / "complete.json"
    return result


def _rewrite_attempt_document(proof, name: str, update):
    path = proof.attempt_dir / name
    value = json.loads(path.read_text(encoding="utf-8"))
    update(value)
    digest = proof.ev._atomic_write_json(path, value)
    complete = json.loads(proof.complete_path.read_text(encoding="utf-8"))
    key = "runtime_result_sha256" if name == "runtime_result.json" else "evaluation_sha256"
    complete[key] = digest
    proof.ev._atomic_write_json(proof.complete_path, complete)


def _runtime_retry_proof(
    monkeypatch, tmp_path: Path, *, candidate_mode: str = "explicit-options"
):
    ev = retry_runtime.evaluator
    candidate = ev.parse_candidate_text(
        _candidate_text(
            options=(
                {"final_terminal_pushforward_policy": "strict"}
                if candidate_mode == "explicit-options"
                else []
            ),
            candidate_mode=candidate_mode,
        )
    )
    slot_root = tmp_path / "slot-0"
    control_repo = slot_root / "control"
    candidate_repo = slot_root / "candidate"
    results = slot_root / "results"
    for repo in (control_repo, candidate_repo):
        repo.mkdir(parents=True)
        (repo / "env.sh").write_text("same env\n", encoding="utf-8")
    results.mkdir()
    slot = ev.Slot(slot_root, control_repo, candidate_repo, results, None)

    artifacts_by_repo = {}
    for repo, binary_data in (
        (control_repo, b"control-binary"),
        (candidate_repo, b"candidate-binary"),
    ):
        binary = repo / "emu"
        image = repo / "image"
        nemu = repo / "nemu"
        binary.write_bytes(binary_data)
        image.write_bytes(b"same-image")
        nemu.write_bytes(b"same-nemu")
        artifacts_by_repo[repo] = (binary, image, nemu)

    control_generated = "c" * 64
    candidate_generated = "d" * 64
    monkeypatch.setattr(ev, "_verify_pinned_repo", lambda *_args: None)
    monkeypatch.setattr(
        ev, "_artifact_paths", lambda repo, **_kwargs: artifacts_by_repo[repo]
    )
    monkeypatch.setattr(
        ev,
        "generated_fingerprint",
        lambda repo: control_generated if repo == control_repo else candidate_generated,
    )
    monkeypatch.setattr(
        ev,
        "_build_config_fingerprint",
        lambda options, *, jobs: "a" * 64 if not options else "b" * 64,
    )
    monkeypatch.setattr(ev, "_toolchain_fingerprint", lambda *_args: "e" * 64)
    monkeypatch.setattr(
        ev,
        "_build_variant",
        lambda *_args, **_kwargs: pytest.fail("runtime-only retry attempted a build"),
    )

    control_binary, control_image, control_nemu = artifacts_by_repo[control_repo]
    control = ev.BuildArtifacts(
        name="control",
        repo=control_repo,
        binary=control_binary,
        image=control_image,
        nemu=control_nemu,
        generated_fingerprint=control_generated,
        build_log=results / "control_build.log",
        build_config_fingerprint="a" * 64,
        toolchain_fingerprint="e" * 64,
    )
    ev._atomic_write_json(
        results / "control_artifacts.json",
        {
            "schema_version": ev.CONTROL_MARKER_SCHEMA_VERSION,
            "name": "control",
            "repo": str(control_repo),
            "binary": str(control_binary),
            "image": str(control_image),
            "nemu": str(control_nemu),
            "generated_fingerprint": control_generated,
            "build_log": str(results / "control_build.log"),
            "build_config_fingerprint": "a" * 64,
            "toolchain_fingerprint": "e" * 64,
            "parent_commit": ev.PINNED_PARENT_COMMIT,
            "wolvrix_commit": ev.PINNED_WOLVRIX_COMMIT,
            "artifact_sha256": {
                "binary": ev._sha256_file(control_binary),
                "image": ev._sha256_file(control_image),
                "nemu": ev._sha256_file(control_nemu),
            },
        },
    )

    patch_path = slot_root / "candidate.patch"
    patch_path.write_text(candidate.patch, encoding="utf-8")
    candidate_binary, candidate_image, candidate_nemu = artifacts_by_repo[candidate_repo]
    candidate_build_config = "b" * 64 if candidate.enable_options else "a" * 64
    enabled = ev.BuildArtifacts(
        name=ev._candidate_artifact_name(candidate),
        repo=candidate_repo,
        binary=candidate_binary,
        image=candidate_image,
        nemu=candidate_nemu,
        generated_fingerprint=candidate_generated,
        build_log=results / f"{ev._candidate_artifact_name(candidate)}_build.log",
        build_config_fingerprint=candidate_build_config,
        toolchain_fingerprint="e" * 64,
    )
    enabled = ev._write_candidate_proof(
        candidate,
        slot,
        control,
        enabled,
        control_artifact_sha256=ev._artifact_sha256(control),
        control_env_sha256=ev._sha256_file(control_repo / "env.sh"),
    )
    proof = SimpleNamespace(
        ev=ev,
        candidate=candidate,
        slot=slot,
        control=control,
        enabled=enabled,
        control_repo=control_repo,
        candidate_repo=candidate_repo,
        control_binary=control_binary,
        candidate_binary=candidate_binary,
        candidate_image=candidate_image,
        candidate_env=candidate_repo / "env.sh",
        patch_path=patch_path,
        proof_path=ev._candidate_proof_path(slot, candidate),
    )
    _publish_retry_attempt(proof)
    return proof


def test_runtime_only_retry_reuses_complete_proof_without_building(
    monkeypatch, tmp_path: Path
):
    proof = _runtime_retry_proof(monkeypatch, tmp_path)

    control, candidate = retry_runtime.prepare_reused_artifacts(
        proof.candidate, proof.slot
    )

    assert control.name == "control"
    assert candidate.name == proof.ev._candidate_artifact_name(proof.candidate)
    assert candidate.generated_fingerprint == "d" * 64
    marker = json.loads(proof.proof_path.read_text(encoding="utf-8"))
    assert set(marker) == retry_runtime._CANDIDATE_PROOF_FIELDS
    assert marker["schema_version"] == proof.ev.CANDIDATE_PROOF_SCHEMA_VERSION
    assert marker["proof_version"] == proof.ev.CANDIDATE_PROOF_VERSION
    assert marker["candidate_digest"] == proof.candidate.digest
    assert marker["candidate_mode"] == proof.candidate.candidate_mode
    assert marker["artifacts"]["binary"]["sha256"] == proof.ev._sha256_file(
        proof.candidate_binary
    )
    assert marker["env_sh"]["candidate_sha256"] == marker["env_sh"]["control_sha256"]
    assert candidate.candidate_proof_sha256 == proof.ev._sha256_file(proof.proof_path)


def test_runtime_only_retry_reuses_default_path_proof(monkeypatch, tmp_path: Path):
    proof = _runtime_retry_proof(
        monkeypatch, tmp_path, candidate_mode="default-path"
    )

    control, candidate = retry_runtime.prepare_reused_artifacts(
        proof.candidate, proof.slot
    )

    assert control.name == "control"
    assert candidate.name == proof.ev._candidate_artifact_name(proof.candidate)
    assert proof.candidate.candidate_mode == "default-path"
    assert candidate.build_config_fingerprint == control.build_config_fingerprint


@pytest.mark.parametrize(
    ("artifact_name", "message"),
    (("image", "image SHA-256 differ"), ("nemu", "NEMU SHA-256 differ")),
)
def test_full_candidate_proof_rejects_workload_or_reference_drift_before_publish(
    monkeypatch, tmp_path: Path, artifact_name: str, message: str
):
    proof = _runtime_retry_proof(monkeypatch, tmp_path)
    control_sha256 = proof.ev._artifact_sha256(proof.control)
    control_env_sha256 = proof.ev._sha256_file(proof.control_repo / "env.sh")
    proof.proof_path.unlink()
    getattr(proof.enabled, artifact_name).write_bytes(b"candidate-owned-drift")

    with pytest.raises(proof.ev.CandidateError, match=message):
        proof.ev._write_candidate_proof(
            proof.candidate,
            proof.slot,
            proof.control,
            proof.enabled,
            control_artifact_sha256=control_sha256,
            control_env_sha256=control_env_sha256,
        )

    assert not proof.proof_path.exists()


def test_full_candidate_proof_rejects_coordinated_control_marker_mutation(
    monkeypatch, tmp_path: Path
):
    proof = _runtime_retry_proof(monkeypatch, tmp_path)
    control_sha256 = proof.ev._artifact_sha256(proof.control)
    control_env_sha256 = proof.ev._sha256_file(proof.control_repo / "env.sh")
    proof.proof_path.unlink()
    proof.control_binary.write_bytes(b"coordinated-control-mutation")
    marker_path = proof.slot.results_dir / "control_artifacts.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifact_sha256"]["binary"] = proof.ev._sha256_file(
        proof.control_binary
    )
    proof.ev._atomic_write_json(marker_path, marker)

    with pytest.raises(proof.ev.InfrastructureError, match="changed during"):
        proof.ev._write_candidate_proof(
            proof.candidate,
            proof.slot,
            proof.control,
            proof.enabled,
            control_artifact_sha256=control_sha256,
            control_env_sha256=control_env_sha256,
        )

    assert not proof.proof_path.exists()


def test_runtime_only_retry_rejects_digest_patch_and_fingerprint_mismatches(
    monkeypatch, tmp_path: Path
):
    proof = _runtime_retry_proof(monkeypatch, tmp_path)
    _rewrite_attempt_document(
        proof, "evaluation.json", lambda value: value.__setitem__("candidate_digest", "0" * 16)
    )
    with pytest.raises(proof.ev.InfrastructureError, match="candidate_digest"):
        retry_runtime.prepare_reused_artifacts(proof.candidate, proof.slot)

    proof = _runtime_retry_proof(monkeypatch, tmp_path / "patch")
    proof.patch_path.write_text("not the requested patch\n", encoding="utf-8")
    with pytest.raises(proof.ev.CandidateError, match="does not exactly match"):
        retry_runtime.prepare_reused_artifacts(proof.candidate, proof.slot)

    proof = _runtime_retry_proof(monkeypatch, tmp_path / "fingerprint")
    marker = json.loads(proof.proof_path.read_text(encoding="utf-8"))
    marker["generated_fingerprint"] = "e" * 64
    proof.ev._atomic_write_json(proof.proof_path, marker)
    with pytest.raises(proof.ev.InfrastructureError, match="proof binding differs"):
        retry_runtime.prepare_reused_artifacts(proof.candidate, proof.slot)

    proof = _runtime_retry_proof(monkeypatch, tmp_path / "mode")
    marker = json.loads(proof.proof_path.read_text(encoding="utf-8"))
    marker["candidate_mode"] = "default-path"
    proof.ev._atomic_write_json(proof.proof_path, marker)
    with pytest.raises(proof.ev.InfrastructureError, match="candidate_mode"):
        retry_runtime.prepare_reused_artifacts(proof.candidate, proof.slot)


def test_runtime_only_retry_binds_candidate_mode_in_attempt_documents(
    monkeypatch, tmp_path: Path
):
    proof = _runtime_retry_proof(monkeypatch, tmp_path)
    _rewrite_attempt_document(
        proof,
        "evaluation.json",
        lambda value: value.__setitem__("candidate_mode", "default-path"),
    )
    with pytest.raises(proof.ev.InfrastructureError, match="candidate_mode"):
        retry_runtime.prepare_reused_artifacts(proof.candidate, proof.slot)


def test_runtime_only_retry_rejects_pinned_control_sha_and_input_failures(
    monkeypatch, tmp_path: Path
):
    proof = _runtime_retry_proof(monkeypatch, tmp_path / "pinned")
    monkeypatch.setattr(
        proof.ev,
        "_verify_pinned_repo",
        lambda *_args: (_ for _ in ()).throw(proof.ev.InfrastructureError("not pinned")),
    )
    with pytest.raises(proof.ev.InfrastructureError, match="not pinned"):
        retry_runtime.prepare_reused_artifacts(proof.candidate, proof.slot)

    proof = _runtime_retry_proof(monkeypatch, tmp_path / "control-sha")
    proof.control_binary.write_bytes(b"mutated-control")
    with pytest.raises(proof.ev.InfrastructureError, match="artifact_sha256"):
        retry_runtime.prepare_reused_artifacts(proof.candidate, proof.slot)

    proof = _runtime_retry_proof(monkeypatch, tmp_path / "inputs")
    proof.candidate_image.write_bytes(b"different-image")
    with pytest.raises(proof.ev.InfrastructureError, match="image SHA-256 differs"):
        retry_runtime.prepare_reused_artifacts(proof.candidate, proof.slot)


def test_runtime_only_retry_rejects_coordinated_control_and_marker_replacement(
    monkeypatch, tmp_path: Path
):
    proof = _runtime_retry_proof(monkeypatch, tmp_path)
    proof.control_binary.write_bytes(b"replacement-control-binary")
    marker_path = proof.slot.results_dir / "control_artifacts.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifact_sha256"]["binary"] = proof.ev._sha256_file(
        proof.control_binary
    )
    proof.ev._atomic_write_json(marker_path, marker)

    with pytest.raises(proof.ev.InfrastructureError, match="control_identity"):
        retry_runtime.prepare_reused_artifacts(proof.candidate, proof.slot)


def test_runtime_only_retry_rejects_candidate_binary_symlink_and_env_changes(
    monkeypatch, tmp_path: Path
):
    proof = _runtime_retry_proof(monkeypatch, tmp_path / "binary")
    proof.candidate_binary.write_bytes(b"mutated-binary")
    with pytest.raises(proof.ev.InfrastructureError, match="binary SHA-256 differs"):
        retry_runtime.prepare_reused_artifacts(proof.candidate, proof.slot)

    proof = _runtime_retry_proof(monkeypatch, tmp_path / "symlink")
    proof.candidate_binary.unlink()
    proof.candidate_binary.symlink_to(proof.control_binary)
    with pytest.raises(proof.ev.InfrastructureError, match="non-regular artifact"):
        retry_runtime.prepare_reused_artifacts(proof.candidate, proof.slot)

    proof = _runtime_retry_proof(monkeypatch, tmp_path / "env")
    proof.candidate_env.unlink()
    proof.candidate_env.symlink_to(proof.control_repo / "env.sh")
    with pytest.raises(proof.ev.InfrastructureError, match="candidate env.sh"):
        retry_runtime.prepare_reused_artifacts(proof.candidate, proof.slot)


def test_runtime_only_retry_requires_strict_manifest_schema_and_flag_types(
    monkeypatch, tmp_path: Path
):
    proof = _runtime_retry_proof(monkeypatch, tmp_path / "flags")
    _rewrite_attempt_document(
        proof,
        "evaluation.json",
        lambda value: value.__setitem__("infrastructure_retry", 1),
    )
    with pytest.raises(proof.ev.CandidateError, match="evaluation.infrastructure_retry"):
        retry_runtime.prepare_reused_artifacts(proof.candidate, proof.slot)

    proof = _runtime_retry_proof(monkeypatch, tmp_path / "schema")
    _rewrite_attempt_document(
        proof,
        "runtime_result.json",
        lambda value: value.__setitem__("schema_version", True),
    )
    with pytest.raises(proof.ev.InfrastructureError, match="schema_version"):
        retry_runtime.prepare_reused_artifacts(proof.candidate, proof.slot)

    proof = _runtime_retry_proof(monkeypatch, tmp_path / "proof-schema")
    marker = json.loads(proof.proof_path.read_text(encoding="utf-8"))
    marker["schema_version"] = True
    proof.ev._atomic_write_json(proof.proof_path, marker)
    with pytest.raises(proof.ev.InfrastructureError, match="malformed fields"):
        retry_runtime.prepare_reused_artifacts(proof.candidate, proof.slot)


def test_runtime_only_retry_uses_immutable_attempt_when_latest_publish_is_interrupted(
    monkeypatch, tmp_path: Path
):
    proof = _runtime_retry_proof(monkeypatch, tmp_path)
    digest = proof.candidate.digest[:16]
    (proof.slot.results_dir / f"runtime_result_{digest}.json").write_text(
        "{}\n", encoding="utf-8"
    )
    incomplete = (
        proof.slot.results_dir
        / "attempts"
        / proof.candidate.digest
        / "99999999999999999999-interrupted"
    )
    incomplete.mkdir()
    (incomplete / "runtime_result.json").write_text("{}\n", encoding="utf-8")

    control, enabled = retry_runtime.prepare_reused_artifacts(
        proof.candidate, proof.slot
    )

    assert control.name == "control"
    assert enabled.candidate_proof_id == proof.enabled.candidate_proof_id


def test_alias_publish_failure_leaves_no_complete_and_preserves_prior_retry(
    monkeypatch, tmp_path: Path
):
    proof = _runtime_retry_proof(monkeypatch, tmp_path)
    attempts_root = proof.slot.results_dir / "attempts" / proof.candidate.digest
    existing = set(attempts_root.iterdir())
    original_atomic_write = proof.ev._atomic_write_json
    failing_alias = (
        proof.slot.results_dir / f"evaluation_{proof.candidate.digest[:16]}.json"
    )

    def fail_second_alias(path, value):
        if path == failing_alias:
            raise OSError("injected compatibility alias failure")
        return original_atomic_write(path, value)

    monkeypatch.setattr(proof.ev, "_atomic_write_json", fail_second_alias)
    with pytest.raises(OSError, match="alias failure"):
        _publish_retry_attempt(proof, error="incomplete retry")
    created = set(attempts_root.iterdir()) - existing
    assert len(created) == 1
    assert not (next(iter(created)) / "complete.json").exists()

    monkeypatch.setattr(proof.ev, "_atomic_write_json", original_atomic_write)
    control, enabled = retry_runtime.prepare_reused_artifacts(
        proof.candidate, proof.slot
    )
    assert control.name == "control"
    assert enabled.candidate_proof_id == proof.enabled.candidate_proof_id


def test_runtime_only_retry_accepts_latest_retry_of_retry(monkeypatch, tmp_path: Path):
    proof = _runtime_retry_proof(monkeypatch, tmp_path)
    second = _publish_retry_attempt(proof, error="second retry")

    _runtime, evaluation = retry_runtime._load_latest_complete_attempt(
        proof.candidate,
        proof.slot,
        proof.enabled.candidate_proof_id,
        proof.enabled.candidate_proof_sha256,
    )

    assert evaluation["attempt_id"] == second["attempt_id"]
    assert evaluation["error"] == "second retry"


def test_attempt_publish_requires_proof_for_noncontrol_and_uses_control_sentinel(
    monkeypatch, tmp_path: Path
):
    proof = _runtime_retry_proof(monkeypatch, tmp_path)
    attempts_root = proof.slot.results_dir / "attempts" / proof.candidate.digest
    before = set(attempts_root.iterdir())
    unproven = SimpleNamespace(candidate_proof_id="", candidate_proof_sha256="")
    with pytest.raises(proof.ev.InfrastructureError, match="valid candidate proof id"):
        proof.ev._publish_evaluation_attempt(
            proof.slot,
            proof.candidate,
            unproven,
            {"valid": False},
            {"valid_candidate": 0},
        )
    assert set(attempts_root.iterdir()) == before

    control_candidate = proof.ev.parse_candidate_text(
        json.dumps(
            {
                "schema_version": 2,
                "candidate_mode": "control",
                "hypothesis": "control",
                "evidence": ["control"],
                "patch": "",
                "enable_options": [],
            }
        )
    )
    published = proof.ev._publish_evaluation_attempt(
        proof.slot,
        control_candidate,
        unproven,
        {"valid": False},
        {"valid_candidate": 0},
    )
    assert published["candidate_proof_id"] == proof.ev.CONTROL_CANDIDATE_PROOF_ID
    assert (
        published["candidate_proof_sha256"]
        == proof.ev.CONTROL_CANDIDATE_PROOF_SHA256
    )


def _existing_retry_slot(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir(parents=True)
    root = tmp_path / "slots"
    root.mkdir()
    namespace = retry_runtime.hashlib.sha256(
        f"{source.resolve()}\0{retry_runtime.evaluator.PINNED_PARENT_COMMIT}\0"
        f"{retry_runtime.evaluator.PINNED_WOLVRIX_COMMIT}".encode("utf-8")
    ).hexdigest()[:16]
    slot = root / namespace / "slot-0"
    (slot / "control").mkdir(parents=True)
    (slot / "candidate").mkdir()
    (slot / "results").mkdir()
    (slot / "lock").write_text("", encoding="utf-8")
    return source, root, slot


def test_runtime_only_lock_rejects_symlink_and_nonregular_lock(monkeypatch, tmp_path: Path):
    source, root, slot = _existing_retry_slot(tmp_path / "symlink")
    target = slot / "lock-target"
    target.write_text("", encoding="utf-8")
    (slot / "lock").unlink()
    (slot / "lock").symlink_to(target)
    monkeypatch.setenv("GRHSIM_SLOT_LOCK_TIMEOUT", "0")
    with pytest.raises(retry_runtime.evaluator.InfrastructureError):
        with retry_runtime.acquire_existing_slot(source, root):
            pass

    source, root, slot = _existing_retry_slot(tmp_path / "directory")
    (slot / "lock").unlink()
    (slot / "lock").mkdir()
    with pytest.raises(retry_runtime.evaluator.InfrastructureError):
        with retry_runtime.acquire_existing_slot(source, root):
            pass


def test_runtime_only_lock_rejects_symlinked_root_and_releases_after_context(
    monkeypatch, tmp_path: Path
):
    source, real_root, _slot = _existing_retry_slot(tmp_path / "root")
    linked_root = tmp_path / "linked-slots"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(retry_runtime.evaluator.InfrastructureError, match="symlink"):
        with retry_runtime.acquire_existing_slot(source, linked_root):
            pass

    monkeypatch.setenv("GRHSIM_SLOT_LOCK_TIMEOUT", "0")
    with retry_runtime.acquire_existing_slot(source, real_root):
        with pytest.raises(retry_runtime.evaluator.InfrastructureError, match="timed out"):
            with retry_runtime.acquire_existing_slot(source, real_root):
                pass
    with retry_runtime.acquire_existing_slot(source, real_root) as slot:
        assert slot.root == _slot


def test_runtime_only_lock_detects_unlink_recreate_after_flock(monkeypatch, tmp_path: Path):
    source, root, slot = _existing_retry_slot(tmp_path)
    lock_path = slot / "lock"
    original_flock = retry_runtime.fcntl.flock
    replaced = False

    def replace_after_lock(descriptor, operation):
        nonlocal replaced
        result = original_flock(descriptor, operation)
        if operation & retry_runtime.fcntl.LOCK_EX and not replaced:
            replaced = True
            lock_path.unlink()
            lock_path.write_text("replacement", encoding="utf-8")
        return result

    monkeypatch.setattr(retry_runtime.fcntl, "flock", replace_after_lock)
    with pytest.raises(retry_runtime.evaluator.InfrastructureError, match="replaced"):
        with retry_runtime.acquire_existing_slot(source, root):
            pass

    monkeypatch.setattr(retry_runtime.fcntl, "flock", original_flock)
    with retry_runtime.acquire_existing_slot(source, root) as acquired:
        assert acquired.root == slot


def test_evaluate_accepts_explicit_slot_acquirer(monkeypatch, tmp_path: Path):
    program = tmp_path / "candidate.txt"
    program.write_text(_candidate_text(), encoding="utf-8")
    control = _artifacts(tmp_path, "control-explicit")
    candidate_artifacts = _artifacts(tmp_path, "candidate-explicit")
    slot = evaluator.Slot(
        tmp_path, control.repo, candidate_artifacts.repo, tmp_path, None
    )
    acquired = []

    @contextmanager
    def explicit_acquirer(source_repo, slot_root):
        acquired.append((source_repo, slot_root))
        yield slot

    def default_acquirer(*_args, **_kwargs):
        pytest.fail("default slot acquirer was used")

    monkeypatch.setattr(evaluator, "acquire_slot", default_acquirer)
    metrics = evaluator.evaluate(
        str(program),
        runtime_fn=lambda *_args, **_kwargs: {
            "valid": True,
            "control_walltime_ms": 100.0,
            "candidate_walltime_ms": 101.0,
            "samples": [],
            "diagnostics": {},
        },
        artifact_preparer=lambda *_args: (control, candidate_artifacts),
        slot_acquirer=explicit_acquirer,
        source_repo=tmp_path / "source",
        slot_root=tmp_path / "slots",
    )

    assert metrics["valid_candidate"] == 1
    assert acquired == [(tmp_path / "source", tmp_path / "slots")]


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


def _write_v2_checkpoint_seed(path: Path, *, pin_override=None) -> str:
    text = _candidate_text()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    parent_pin, wolvrix_pin = pin_override or (
        evaluator.PINNED_PARENT_COMMIT[:12],
        evaluator.PINNED_WOLVRIX_COMMIT[:12],
    )
    (path.parent / "nodes.json").write_text(
        json.dumps(
            [
                {
                    "code": text,
                    "metrics": {
                        "parent_commit": parent_pin,
                        "wolvrix_commit": wolvrix_pin,
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    return text


def test_launcher_uses_fixed_model_serial_budget_and_only_secret_file_paths(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setenv("GRHSIM_VERIFY_DEFAULT_OFF", "0")
    monkeypatch.setenv("GRHSIM_RUN_FUNCTION_GATES", "0")
    monkeypatch.setenv("GRHSIM_RUN_FOCUSED_TESTS", "0")
    target = tmp_path / "target"
    (target / ".git").mkdir(parents=True)
    (target / "env.sh").write_text("true\n", encoding="utf-8")
    config = tmp_path / "config.kimi.toml"
    auth = tmp_path / "auth.kimi.json"
    config.write_text("model_provider = 'kimi'\n", encoding="utf-8")
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
    assert "--model k3" in rendered
    assert "--reasoning-effort ultra" in rendered
    assert "--max-valid-evaluations 8" in rendered
    assert "--max-generations 16" in rendered
    assert "--eval-concurrency 1" in rendered
    assert "--gen-concurrency 1" in rendered
    assert "--skip-preflight" in rendered
    local_schema_index = command.index("--codex-local-validation-schema")
    assert command[local_schema_index + 1] == str(
        launcher.LOCAL_VALIDATION_SCHEMA.resolve()
    )
    provider_schema_index = command.index("--codex-output-schema")
    assert command[provider_schema_index + 1] == str(
        (TASK_ROOT / "candidate.schema.json").resolve()
    )
    output_mode_index = command.index("--codex-output-mode")
    assert command[output_mode_index + 1] == "local-json"
    tool_choice_index = command.index("--codex-tool-choice-mode")
    assert command[tool_choice_index + 1] == "required-first"
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
    _write_v2_checkpoint_seed(best_program)
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


def test_launcher_rejects_legacy_seed_and_pre_rwa_best_program(tmp_path: Path):
    target = tmp_path / "target"
    (target / ".git").mkdir(parents=True)
    (target / "env.sh").write_text("true\n", encoding="utf-8")
    config = tmp_path / "config.mjy.toml"
    auth = tmp_path / "auth.mjy.json"
    config.write_text("model_provider = 'mjy'\n", encoding="utf-8")
    auth.write_text("{}\n", encoding="utf-8")
    legacy = tmp_path / "legacy.txt"
    legacy.write_text(
        _candidate_text().replace('"schema_version": 2', '"schema_version": 1'),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        target_repo=target,
        codex_config=config,
        codex_auth=auth,
        init_program=legacy,
        output_path=tmp_path / "checkpoints",
        slot_root=tmp_path / "slots",
        resume=None,
        max_proposals=16,
        valid_target=8,
        eval_timeout=21_600.0,
        llm_timeout=3_000.0,
        max_tokens=32_768,
    )
    with pytest.raises(SystemExit, match="schema v2"):
        launcher.build_command(args)

    wrong_pin = tmp_path / "old" / "db_state_010203" / "best_program.txt"
    _write_v2_checkpoint_seed(
        wrong_pin,
        pin_override=(
            OLD_PRE_RWA_PARENT_COMMIT[:12],
            OLD_PRE_RWA_WOLVRIX_COMMIT[:12],
        ),
    )
    args.init_program = wrong_pin
    with pytest.raises(SystemExit, match="evaluator pins differ"):
        launcher.build_command(args)


def test_launcher_rejects_legacy_or_pre_rwa_resume(tmp_path: Path):
    target = tmp_path / "target"
    (target / ".git").mkdir(parents=True)
    (target / "env.sh").write_text("true\n", encoding="utf-8")
    config = tmp_path / "config.mjy.toml"
    auth = tmp_path / "auth.mjy.json"
    config.write_text("model_provider = 'mjy'\n", encoding="utf-8")
    auth.write_text("{}\n", encoding="utf-8")
    state = tmp_path / "instance" / "db_state_010203"
    best = state / "best_program.txt"
    _write_v2_checkpoint_seed(
        best,
        pin_override=(
            OLD_PRE_RWA_PARENT_COMMIT[:12],
            OLD_PRE_RWA_WOLVRIX_COMMIT[:12],
        ),
    )
    args = SimpleNamespace(
        target_repo=target,
        codex_config=config,
        codex_auth=auth,
        init_program=launcher.DEFAULT_INIT_PROGRAM,
        output_path=tmp_path / "checkpoints",
        slot_root=tmp_path / "slots",
        resume=state.parent,
        max_proposals=16,
        valid_target=8,
        eval_timeout=21_600.0,
        llm_timeout=3_000.0,
        max_tokens=32_768,
    )
    with pytest.raises(SystemExit, match="evaluator pins differ"):
        launcher.build_command(args)

    _write_v2_checkpoint_seed(best)
    best.write_text(
        best.read_text(encoding="utf-8").replace(
            '"schema_version": 2', '"schema_version": 1'
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="schema v2"):
        launcher.build_command(args)


def test_launcher_resume_uses_the_exact_validated_state_and_seed(
    monkeypatch, tmp_path: Path
):
    target = tmp_path / "target"
    (target / ".git").mkdir(parents=True)
    (target / "env.sh").write_text("true\n", encoding="utf-8")
    config = tmp_path / "config.mjy.toml"
    auth = tmp_path / "auth.mjy.json"
    config.write_text("model_provider = 'mjy'\n", encoding="utf-8")
    auth.write_text("{}\n", encoding="utf-8")
    instance = tmp_path / "instance"
    selected_state = instance / "db_state_010203"
    selected_seed = selected_state / "best_program.txt"
    _write_v2_checkpoint_seed(selected_seed)
    args = SimpleNamespace(
        target_repo=target,
        codex_config=config,
        codex_auth=auth,
        init_program=launcher.DEFAULT_INIT_PROGRAM,
        output_path=tmp_path / "checkpoints",
        slot_root=tmp_path / "slots",
        resume=instance,
        max_proposals=16,
        valid_target=8,
        eval_timeout=21_600.0,
        llm_timeout=3_000.0,
        max_tokens=32_768,
    )
    original_validate = launcher._validate_checkpoint_contract

    def validate_then_publish_newer(*validate_args, **validate_kwargs):
        validated = original_validate(*validate_args, **validate_kwargs)
        newer_seed = instance / "db_state_999999" / "best_program.txt"
        _write_v2_checkpoint_seed(newer_seed)
        return validated

    monkeypatch.setattr(
        launcher, "_validate_checkpoint_contract", validate_then_publish_newer
    )

    command, _environment = launcher.build_command(args)
    resume_index = command.index("--resume")
    init_index = command.index("--init-program")

    assert command[resume_index + 1] == str(selected_state.resolve())
    assert command[init_index + 1] == str(selected_seed.resolve())
    assert command[resume_index + 1] != str(instance.resolve())


def test_launcher_cli_defaults_to_dataset_initial_program(monkeypatch, capsys):
    captured = {}

    def fake_build_command(args):
        captured["init_program"] = args.init_program
        captured["codex_config"] = args.codex_config
        captured["codex_auth"] = args.codex_auth
        return ["true"], {}

    monkeypatch.setattr(launcher, "build_command", fake_build_command)
    monkeypatch.setattr(sys, "argv", ["launcher.py", "--dry-run"])

    assert launcher.main() == 0
    assert captured["init_program"] == launcher.DEFAULT_INIT_PROGRAM
    assert captured["codex_config"] == Path("~/.codex/config.kimi.toml").expanduser()
    assert captured["codex_auth"] == Path("~/.codex/auth.kimi.json").expanduser()
    assert capsys.readouterr().out.strip() == "true"


def _capability_payload(digest: str) -> dict:
    payload = json.loads(
        evaluator._extract_candidate_payload(
            _candidate_text(
                options=[],
                candidate_mode="default-path",
                evidence=[f"{launcher.PREFLIGHT_EVIDENCE_PREFIX}{digest}"],
                patch=launcher.PREFLIGHT_SMOKE_PATCH,
            )
        )
    )
    payload["hypothesis"] = launcher.PREFLIGHT_SMOKE_HYPOTHESIS
    return payload


def test_launcher_probe_digest_reads_the_pinned_tracked_blob(monkeypatch, tmp_path: Path):
    parent_pin = "1" * 40
    wolvrix_pin = "2" * 40
    blob = b"tracked pinned source\n"
    nonce = "3" * 64
    calls = []

    def fake_git_bytes(repo, arguments, label):
        calls.append((repo, arguments, label))
        if arguments[-1] == f"{parent_pin}^{{commit}}":
            return f"{parent_pin}\n".encode()
        if arguments[-1] == f"{parent_pin}:wolvrix":
            return f"{wolvrix_pin}\n".encode()
        if arguments[-1] == f"{wolvrix_pin}^{{commit}}":
            return f"{wolvrix_pin}\n".encode()
        if arguments[-1] == f"{wolvrix_pin}:{launcher.PREFLIGHT_PROBE_SUBMODULE_PATH}":
            return blob
        pytest.fail(f"unexpected git probe: {arguments}")

    monkeypatch.setattr(launcher, "_git_bytes", fake_git_bytes)
    contract = SimpleNamespace(
        PINNED_PARENT_COMMIT=parent_pin,
        PINNED_WOLVRIX_COMMIT=wolvrix_pin,
    )

    assert launcher._pinned_probe_digest(
        tmp_path, contract, nonce
    ) == hashlib.sha256(nonce.encode("ascii") + b"\0" + blob).hexdigest()
    assert calls[-1][0] == tmp_path / "wolvrix"
    assert calls[-1][1] == [
        "show",
        f"{wolvrix_pin}:{launcher.PREFLIGHT_PROBE_SUBMODULE_PATH}",
    ]


def test_launcher_checks_capability_patch_in_temporary_pinned_index(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    wolvrix = target / "wolvrix"
    source = wolvrix / "lib" / "example.cpp"
    source.parent.mkdir(parents=True)
    source.write_text("old\n", encoding="utf-8")

    def git(*arguments: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(wolvrix), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=launcher._clean_git_environment(),
        )

    git("init", "-q")
    git("config", "user.email", "simpletes-test@example.invalid")
    git("config", "user.name", "SimpleTES Test")
    git("add", "lib/example.cpp")
    git("commit", "-q", "-m", "pinned")
    pin = git("rev-parse", "HEAD").stdout.decode().strip()
    contract = SimpleNamespace(PINNED_WOLVRIX_COMMIT=pin)
    objects = wolvrix / ".git" / "objects"
    objects_before = {
        path.relative_to(objects): path.read_bytes()
        for path in objects.rglob("*")
        if path.is_file()
    }
    refs_before = git("show-ref").stdout
    index_before = (wolvrix / ".git" / "index").read_bytes()
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "poisoned-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "poisoned-work-tree"))
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", str(tmp_path / "poisoned-objects"))

    assert (
        launcher._check_pinned_patch(
            target,
            contract,
            VALID_PATCH,
            ("lib/example.cpp",),
        )
        is None
    )
    assert source.read_text(encoding="utf-8") == "old\n"
    objects_after = {
        path.relative_to(objects): path.read_bytes()
        for path in objects.rglob("*")
        if path.is_file()
    }
    assert objects_after == objects_before
    assert git("show-ref").stdout == refs_before
    assert (wolvrix / ".git" / "index").read_bytes() == index_before
    assert git("diff", "--quiet").returncode == 0
    assert git("diff", "--cached", "--quiet").returncode == 0

    stale = VALID_PATCH.replace("-old", "-missing")
    assert "does not apply cleanly" in launcher._check_pinned_patch(
        target,
        contract,
        stale,
        ("lib/example.cpp",),
    )


def test_launcher_capability_validator_requires_real_candidate_digest_and_tool_use():
    digest = "a" * 64
    trace = SimpleNamespace(repo_tool_call_count=1, repo_tool_types=("shell",))
    valid = _capability_payload(digest)

    assert (
        launcher._validate_preflight_candidate(
            valid,
            trace,
            evaluator=evaluator,
            expected_digest=digest,
        )
        is None
    )

    wrong_digest = _capability_payload("b" * 64)
    assert "does not match" in launcher._validate_preflight_candidate(
        wrong_digest,
        trace,
        evaluator=evaluator,
        expected_digest=digest,
    )
    assert "did not use" in launcher._validate_preflight_candidate(
        valid,
        SimpleNamespace(repo_tool_call_count=0, repo_tool_types=()),
        evaluator=evaluator,
        expected_digest=digest,
    )

    placeholder = dict(valid)
    placeholder["hypothesis"] = "placeholder"
    assert "placeholder" in launcher._validate_preflight_candidate(
        placeholder,
        trace,
        evaluator=evaluator,
        expected_digest=digest,
    )

    placeholder_evidence = dict(valid)
    placeholder_evidence["evidence"] = [
        f"{launcher.PREFLIGHT_EVIDENCE_PREFIX}{digest}",
        "placeholder",
    ]
    assert "evidence contains placeholder" in launcher._validate_preflight_candidate(
        placeholder_evidence,
        trace,
        evaluator=evaluator,
        expected_digest=digest,
    )

    invalid_mode_shape = dict(valid)
    invalid_mode_shape["candidate_mode"] = "explicit-options"
    assert "not a valid production" in launcher._validate_preflight_candidate(
        invalid_mode_shape,
        trace,
        evaluator=evaluator,
        expected_digest=digest,
    )

    invalid_patch = dict(valid)
    invalid_patch["patch"] = "placeholder\n"
    assert "not a valid production" in launcher._validate_preflight_candidate(
        invalid_patch,
        trace,
        evaluator=evaluator,
        expected_digest=digest,
    )

    missing_newline = dict(valid)
    missing_newline["patch"] = VALID_PATCH.rstrip("\n")
    assert "not a valid production" in launcher._validate_preflight_candidate(
        missing_newline,
        trace,
        evaluator=evaluator,
        expected_digest=digest,
    )


def test_launcher_capability_preflight_is_repo_grounded_and_research_free(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys,
):
    target = tmp_path / "target"
    (target / ".git").mkdir(parents=True)
    (target / "env.sh").write_text("true\n", encoding="utf-8")
    config = tmp_path / "config.kimi.toml"
    auth = tmp_path / "auth.kimi.json"
    config.write_text("model = 'k3'\n", encoding="utf-8")
    auth.write_text('{"OPENAI_API_KEY":"not-read-by-fake"}\n', encoding="utf-8")
    digest = "c" * 64
    nonce = "e" * 64
    payload = _capability_payload(digest)
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.closed = False

        async def capability_probe(self, prompt, *, validate=None):
            captured["prompt"] = prompt
            trace = SimpleNamespace(repo_tool_call_count=2, repo_tool_types=("shell",))
            assert validate is not None
            assert validate(payload, trace) is None
            return SimpleNamespace(
                value=payload,
                trace=trace,
                canonical=json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, indent=2
                ),
            )

        def close(self):
            self.closed = True
            captured["closed"] = True

    import simpletes.llm.codex_exec as codex_exec

    def forbidden(*_args, **_kwargs):
        pytest.fail("capability preflight must not start research or evaluation")

    monkeypatch.setattr(codex_exec, "CodexExecClient", FakeClient)
    monkeypatch.setattr(launcher, "_load_evaluator", lambda: evaluator)
    monkeypatch.setattr(launcher, "_pinned_probe_digest", lambda *_args: digest)
    monkeypatch.setattr(launcher, "_check_pinned_patch", lambda *_args: None)
    monkeypatch.setattr(launcher.secrets, "token_hex", lambda _size: nonce)
    monkeypatch.setattr(launcher, "build_command", forbidden)
    monkeypatch.setattr(evaluator, "evaluate", forbidden)
    args = SimpleNamespace(
        target_repo=target,
        codex_config=config,
        codex_auth=auth,
        preflight_timeout=600.0,
    )

    assert launcher.run_codex_preflight(args) == 0

    assert captured["closed"] is True
    assert captured["kwargs"]["output_schema"] == str(
        (TASK_ROOT / "candidate.schema.json").resolve()
    )
    assert captured["kwargs"]["local_validation_schema"] == str(
        launcher.LOCAL_VALIDATION_SCHEMA.resolve()
    )
    assert captured["kwargs"]["output_mode"] == "local-json"
    assert captured["kwargs"]["tool_choice_mode"] == "required-first"
    assert captured["kwargs"]["timeout"] == 600.0
    assert captured["kwargs"]["max_repair_attempts"] == 0
    assert evaluator.PINNED_WOLVRIX_COMMIT in captured["prompt"]
    assert launcher.PREFLIGHT_PROBE_SUBMODULE_PATH in captured["prompt"]
    assert nonce in captured["prompt"]
    assert digest not in captured["prompt"]
    assert "This is not performance research" in captured["prompt"]
    assert "Research the repository deeply" not in captured["prompt"]
    assert json.dumps(launcher.PREFLIGHT_SMOKE_PATCH) in captured["prompt"]
    assert not (tmp_path / "checkpoints").exists()
    output = capsys.readouterr().out
    assert "capability preflight passed" in output
    assert "repo_tool_calls=2" in output


def test_launcher_capability_failure_includes_backend_scrubbed_diagnostic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    from simpletes.llm.types import LLMCallError

    target = tmp_path / "target"
    (target / ".git").mkdir(parents=True)
    (target / "env.sh").write_text("true\n", encoding="utf-8")
    config = tmp_path / "config.kimi.toml"
    auth = tmp_path / "auth.kimi.json"
    config.write_text("model = 'k3'\n", encoding="utf-8")
    auth.write_text('{"OPENAI_API_KEY":"not-read-by-fake"}\n', encoding="utf-8")
    captured = {}

    class FailingClient:
        def __init__(self, **_kwargs):
            pass

        async def capability_probe(self, _prompt, *, validate=None):
            assert validate is not None
            raise LLMCallError(
                model="k3",
                api_base=None,
                error_type="SemanticValidationError",
                message="bounded scrubbed semantic diagnostic",
            )

        def close(self):
            captured["closed"] = True

    import simpletes.llm.codex_exec as codex_exec

    monkeypatch.setattr(codex_exec, "CodexExecClient", FailingClient)
    monkeypatch.setattr(launcher, "_load_evaluator", lambda: evaluator)
    monkeypatch.setattr(launcher, "_pinned_probe_digest", lambda *_args: "d" * 64)
    monkeypatch.setattr(launcher.secrets, "token_hex", lambda _size: "f" * 64)
    args = SimpleNamespace(
        target_repo=target,
        codex_config=config,
        codex_auth=auth,
        preflight_timeout=600.0,
    )

    with pytest.raises(SystemExit) as raised:
        launcher.run_codex_preflight(args)

    message = str(raised.value)
    assert message == (
        "Codex capability preflight failed (SemanticValidationError): "
        "bounded scrubbed semantic diagnostic"
    )
    assert captured["closed"] is True


def test_launcher_preflight_only_never_builds_research_command(monkeypatch, capsys):
    captured = {}

    def fake_preflight(args):
        captured["model"] = launcher.MODEL
        captured["config"] = args.codex_config
        captured["auth"] = args.codex_auth
        captured["timeout"] = args.preflight_timeout
        return 0

    def forbidden_build(_args):
        pytest.fail("preflight-only must not construct a research command")

    monkeypatch.setattr(launcher, "run_codex_preflight", fake_preflight)
    monkeypatch.setattr(launcher, "build_command", forbidden_build)
    monkeypatch.setattr(sys, "argv", ["launcher.py", "--preflight-only"])

    assert launcher.main() == 0
    assert captured["model"] == "k3"
    assert captured["config"] == Path("~/.codex/config.kimi.toml").expanduser()
    assert captured["auth"] == Path("~/.codex/auth.kimi.json").expanduser()
    assert captured["timeout"] == launcher.DEFAULT_PREFLIGHT_TIMEOUT
    assert capsys.readouterr().out == ""


def test_launcher_normal_entry_gates_before_spawning_main(monkeypatch):
    calls = []

    def fake_build(_args):
        calls.append("build")
        return ["main-command"], {"SAFE": "1"}

    def fake_preflight(_args):
        calls.append("capability-preflight")
        return 0

    def fake_run(command, **kwargs):
        calls.append("spawn")
        assert command == ["main-command"]
        assert kwargs["env"] == {"SAFE": "1"}
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launcher, "build_command", fake_build)
    monkeypatch.setattr(launcher, "run_codex_preflight", fake_preflight)
    monkeypatch.setattr(
        launcher, "_launch_guard_digest", lambda _args, _command: "stable"
    )
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["launcher.py"])

    assert launcher.main() == 0
    assert calls == ["build", "capability-preflight", "spawn"]


def test_launcher_capability_failure_never_spawns_main(monkeypatch):
    monkeypatch.setattr(launcher, "build_command", lambda _args: (["main"], {}))
    monkeypatch.setattr(
        launcher, "_launch_guard_digest", lambda _args, _command: "stable"
    )

    def fail_gate(_args):
        raise SystemExit("capability failed")

    def forbidden_spawn(*_args, **_kwargs):
        pytest.fail("failed capability gate must not spawn SimpleTES main")

    monkeypatch.setattr(launcher, "run_codex_preflight", fail_gate)
    monkeypatch.setattr(launcher.subprocess, "run", forbidden_spawn)
    monkeypatch.setattr(sys, "argv", ["launcher.py"])

    with pytest.raises(SystemExit, match="capability failed"):
        launcher.main()


def test_launcher_refuses_spawn_when_inputs_change_during_preflight(monkeypatch):
    guards = iter(["before", "after"])
    calls = []

    monkeypatch.setattr(launcher, "build_command", lambda _args: (["main"], {}))
    monkeypatch.setattr(
        launcher, "_launch_guard_digest", lambda _args, _command: next(guards)
    )
    monkeypatch.setattr(
        launcher, "run_codex_preflight", lambda _args: calls.append("preflight")
    )
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail(
            "changed launch inputs must never spawn research"
        ),
    )
    monkeypatch.setattr(sys, "argv", ["launcher.py"])

    with pytest.raises(SystemExit, match="changed during capability preflight"):
        launcher.main()
    assert calls == ["preflight"]
