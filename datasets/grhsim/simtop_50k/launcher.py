#!/usr/bin/env python3
"""Launch the bounded GrhSIM SimTop-50k SimpleTES research run."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import tempfile
from types import ModuleType
from typing import Any


TASK_ROOT = Path(__file__).resolve().parent
SIMPLETES_ROOT = TASK_ROOT.parents[2]
DEFAULT_TARGET_REPO = SIMPLETES_ROOT.parent / "wolvrix-playground-gsim-calibrate-5"
DEFAULT_CODEX_CONFIG = Path("~/.codex/config.kimi.toml").expanduser()
DEFAULT_CODEX_AUTH = Path("~/.codex/auth.kimi.json").expanduser()
DEFAULT_INIT_PROGRAM = TASK_ROOT / "init_program.txt"
LOCAL_VALIDATION_SCHEMA = TASK_ROOT / "candidate.local.schema.json"
K3_MODEL_CATALOG = TASK_ROOT / "k3_model_catalog.json"
K3_INSTRUCTION_SUFFIX = TASK_ROOT / "instruction.k3.txt"
MODEL = "k3"
REASONING_EFFORT = "ultra"
MAX_PROPOSALS = 64
MAX_EXTENDED_PROPOSALS = 256
NUM_CHAINS = 4
DEFAULT_GEN_CONCURRENCY = NUM_CHAINS
DEFAULT_LLM_TIMEOUT = 10_800.0
DEFAULT_CODEX_MAX_AGENT_THREADS = 3
DEFAULT_CODEX_EXEC_RETRIES = 2
DEFAULT_CODEX_CAPACITY_CONTINUATIONS = 3
DEFAULT_CODEX_TRANSIENT_CONTINUATIONS = 3
DEFAULT_PREFLIGHT_TIMEOUT = 600.0
PREFLIGHT_ATTEMPT_ARTIFACT_ROOT = (
    SIMPLETES_ROOT
    / "checkpoints"
    / "grhsim_simtop_50k"
    / ".preflight_llm_attempts"
)
PREFLIGHT_RUNTIME_HOME_ROOT = (
    SIMPLETES_ROOT / "checkpoints" / "grhsim_simtop_50k" / ".codex_runtime"
)
PREFLIGHT_PROBE_PATH = "wolvrix/lib/emit/grhsim_cpp.cpp"
PREFLIGHT_PROBE_SUBMODULE_PATH = "lib/emit/grhsim_cpp.cpp"
PREFLIGHT_EVIDENCE_PREFIX = "repo_probe_attestation="
PREFLIGHT_SMOKE_HYPOTHESIS = (
    "Capability smoke only: reorder two standard-library includes with no "
    "intended semantic or performance effect."
)
PREFLIGHT_SMOKE_PATCH = (
    "diff --git a/lib/emit/grhsim_cpp.cpp b/lib/emit/grhsim_cpp.cpp\n"
    "--- a/lib/emit/grhsim_cpp.cpp\n"
    "+++ b/lib/emit/grhsim_cpp.cpp\n"
    "@@ -6,7 +6,7 @@\n"
    " \n"
    " #include <algorithm>\n"
    "-#include <atomic>\n"
    " #include <array>\n"
    "+#include <atomic>\n"
    " #include <cctype>\n"
    " #include <chrono>\n"
    " #include <cstdio>\n"
)
_PLACEHOLDER_RE = re.compile(
    r"\s*(?:placeholder|dummy|todo|tbd|unknown|n/?a)\s*[.!]?\s*", re.I
)


def _selected_model(args: argparse.Namespace) -> str:
    """Return the requested model while preserving the dataset's K3 default."""
    model = getattr(args, "model", MODEL)
    if not isinstance(model, str) or not model.strip():
        raise SystemExit("--model must be a non-empty string")
    return model.strip()


def _selected_reasoning_effort(args: argparse.Namespace) -> str:
    effort = getattr(args, "reasoning_effort", REASONING_EFFORT)
    if not isinstance(effort, str) or not effort.strip():
        raise SystemExit("--reasoning-effort must be a non-empty string")
    return effort.strip()


def _selected_codex_exec_retries(args: argparse.Namespace) -> int:
    retries = getattr(
        args, "codex_exec_retries", DEFAULT_CODEX_EXEC_RETRIES
    )
    if type(retries) is not int or not 0 <= retries <= 3:
        raise SystemExit("--codex-exec-retries must be in 0..3")
    return retries


def _selected_codex_capacity_continuations(args: argparse.Namespace) -> int:
    continuations = getattr(
        args,
        "codex_capacity_continuations",
        DEFAULT_CODEX_CAPACITY_CONTINUATIONS,
    )
    if type(continuations) is not int or not 0 <= continuations <= 8:
        raise SystemExit("--codex-capacity-continuations must be in 0..8")
    return continuations


def _selected_codex_transient_continuations(args: argparse.Namespace) -> int:
    continuations = getattr(
        args,
        "codex_transient_continuations",
        DEFAULT_CODEX_TRANSIENT_CONTINUATIONS,
    )
    if type(continuations) is not int or not 0 <= continuations <= 8:
        raise SystemExit("--codex-transient-continuations must be in 0..8")
    return continuations


@dataclass(frozen=True)
class _ValidatedCheckpoint:
    state_dir: Path
    seed_path: Path


def _load_evaluator() -> ModuleType:
    module_name = "_simpletes_grhsim_simtop_launcher_evaluator"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = TASK_ROOT / "evaluator.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load evaluator contract: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path, label: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"{label} must be a regular file: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"cannot read {label}: {error}") from error


def _parse_seed(path: Path, evaluator: ModuleType, label: str) -> Any:
    try:
        return evaluator.parse_candidate_file(path)
    except evaluator.CandidateError as error:
        raise SystemExit(
            f"{label} is incompatible with GrhSIM candidate schema v"
            f"{evaluator.SCHEMA_VERSION}: {error}"
        ) from error


def _checkpoint_state_dir(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise SystemExit(f"{label} must not be a symbolic link: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_dir():
        raise SystemExit(f"{label} must be a directory: {resolved}")
    if resolved.name.startswith("db_state_"):
        return resolved
    states = sorted(
        child
        for child in resolved.iterdir()
        if child.name.startswith("db_state_") and child.is_dir() and not child.is_symlink()
    )
    if not states:
        raise SystemExit(f"{label} contains no db_state_* checkpoint: {resolved}")
    return states[-1]


def _validate_checkpoint_contract(
    state_dir: Path,
    evaluator: ModuleType,
    *,
    expected_candidate: Any | None = None,
) -> _ValidatedCheckpoint:
    best_programs = sorted(state_dir.glob("best_program.*"))
    if len(best_programs) != 1:
        raise SystemExit(
            f"checkpoint must contain exactly one best_program file: {state_dir}"
        )
    best_program = _regular_file(best_programs[0], "checkpoint best program")
    best = _parse_seed(best_program, evaluator, "checkpoint best program")
    if expected_candidate is not None and best.digest != expected_candidate.digest:
        raise SystemExit("checkpoint seed does not match its sibling best_program")

    nodes = _read_json(state_dir / "nodes.json", "checkpoint nodes")
    if not isinstance(nodes, list):
        raise SystemExit("checkpoint nodes must be a JSON array")
    expected_parent = evaluator.PINNED_PARENT_COMMIT[:12]
    expected_wolvrix = evaluator.PINNED_WOLVRIX_COMMIT[:12]
    observed_pins: list[tuple[str, str]] = []
    matched_best = False
    best_pins: list[tuple[str, str]] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        metrics = node.get("metrics")
        if isinstance(metrics, dict):
            parent = metrics.get("parent_commit")
            wolvrix = metrics.get("wolvrix_commit")
            if isinstance(parent, str) and isinstance(wolvrix, str):
                observed_pins.append((parent, wolvrix))
        code = node.get("code")
        if isinstance(code, str):
            try:
                node_candidate = evaluator.parse_candidate_text(code)
            except evaluator.CandidateError:
                continue
            if node_candidate.digest == best.digest:
                matched_best = True
                if isinstance(metrics, dict):
                    parent = metrics.get("parent_commit")
                    wolvrix = metrics.get("wolvrix_commit")
                    if isinstance(parent, str) and isinstance(wolvrix, str):
                        best_pins.append((parent, wolvrix))
    if not matched_best:
        raise SystemExit("checkpoint best program has no matching schema-v2 node")
    if not observed_pins:
        raise SystemExit("checkpoint nodes contain no evaluator pin provenance")
    if not best_pins or any(
        pins != (expected_parent, expected_wolvrix) for pins in best_pins
    ):
        raise SystemExit(
            "checkpoint evaluator pins differ from the current contract for its best program"
        )
    mismatched = sorted(
        {pins for pins in observed_pins if pins != (expected_parent, expected_wolvrix)}
    )
    if mismatched:
        raise SystemExit(
            "checkpoint evaluator pins differ from the current contract: "
            f"expected=({expected_parent},{expected_wolvrix}), observed={mismatched}"
        )
    return _ValidatedCheckpoint(
        state_dir=state_dir.resolve(strict=True),
        seed_path=best_program,
    )


def _regular_file(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise SystemExit(f"{label} must not be a symbolic link: {expanded}")
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise SystemExit(f"{label} must be a regular file: {resolved}")
    return resolved


def _launch_guard_digest(
    args: argparse.Namespace, command: list[str] | None = None
) -> str:
    """Fingerprint runtime inputs so a long preflight cannot gate newer code."""
    paths = {
        *SIMPLETES_ROOT.joinpath("simpletes").rglob("*.py"),
        *TASK_ROOT.glob("*.py"),
        *TASK_ROOT.glob("*.json"),
        *TASK_ROOT.glob("*.txt"),
        SIMPLETES_ROOT / "main.py",
        Path(args.target_repo).expanduser() / "env.sh",
        Path(args.codex_config).expanduser(),
        Path(args.codex_auth).expanduser(),
        Path(args.init_program).expanduser(),
    }
    if command is not None:
        for flag in (
            "--init-program",
            "--evaluator",
            "--instruction",
            "--instruction-suffix",
            "--codex-config",
            "--codex-auth",
            "--codex-output-schema",
            "--codex-local-validation-schema",
        ):
            if flag in command:
                index = command.index(flag)
                if index + 1 >= len(command):
                    raise SystemExit("cannot fingerprint SimpleTES launch inputs")
                paths.add(Path(command[index + 1]))
        if "--resume" in command:
            index = command.index("--resume")
            if index + 1 >= len(command):
                raise SystemExit("cannot fingerprint SimpleTES launch inputs")
            resume_state = Path(command[index + 1])
            if resume_state.is_symlink() or not resume_state.is_dir():
                raise SystemExit("cannot fingerprint SimpleTES launch inputs")
            paths.update(
                path
                for path in resume_state.rglob("*")
                if path.is_file() or path.is_symlink()
            )
    digest = hashlib.sha256()
    try:
        for unresolved in sorted(paths, key=lambda item: str(item)):
            if unresolved.is_symlink():
                raise OSError("guard input is a symbolic link")
            path = unresolved.resolve(strict=True)
            if not path.is_file():
                raise OSError("guard input is not a regular file")
            encoded_path = str(path).encode("utf-8")
            digest.update(len(encoded_path).to_bytes(8, "big"))
            digest.update(encoded_path)
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(len(chunk).to_bytes(8, "big"))
                    digest.update(chunk)
            digest.update((0).to_bytes(8, "big"))
    except (OSError, UnicodeError):
        raise SystemExit("cannot fingerprint SimpleTES launch inputs") from None
    return digest.hexdigest()


def _clean_git_environment(**overrides: str) -> dict[str, str]:
    """Drop caller-controlled Git routing before inspecting pinned objects."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_OPTIONAL_LOCKS": "0",
            **overrides,
        }
    )
    return environment


def _git_bytes(repo: Path, arguments: list[str], label: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_clean_git_environment(),
        )
    except OSError as error:
        raise SystemExit(f"cannot inspect {label} in the target repository") from error
    if completed.returncode != 0:
        raise SystemExit(f"cannot inspect {label} in the target repository")
    return completed.stdout


def _pinned_probe_digest(
    target_repo: Path, evaluator: ModuleType, challenge_nonce: str
) -> str:
    """Bind a fresh challenge to one immutable blob outside the model process."""
    if re.fullmatch(r"[0-9a-f]{64}", challenge_nonce) is None:
        raise SystemExit("preflight challenge nonce must be 64 lowercase hex digits")
    parent = _git_bytes(
        target_repo,
        ["rev-parse", "--verify", f"{evaluator.PINNED_PARENT_COMMIT}^{{commit}}"],
        "pinned parent commit",
    ).decode("ascii", errors="strict").strip()
    if parent != evaluator.PINNED_PARENT_COMMIT:
        raise SystemExit("target repository does not contain the pinned parent commit")
    gitlink = _git_bytes(
        target_repo,
        ["rev-parse", "--verify", f"{parent}:wolvrix"],
        "pinned Wolvrix gitlink",
    ).decode("ascii", errors="strict").strip()
    if gitlink != evaluator.PINNED_WOLVRIX_COMMIT:
        raise SystemExit("pinned parent commit has an unexpected Wolvrix gitlink")

    wolvrix_repo = target_repo / "wolvrix"
    wolvrix_commit = _git_bytes(
        wolvrix_repo,
        ["rev-parse", "--verify", f"{evaluator.PINNED_WOLVRIX_COMMIT}^{{commit}}"],
        "pinned Wolvrix commit",
    ).decode("ascii", errors="strict").strip()
    if wolvrix_commit != evaluator.PINNED_WOLVRIX_COMMIT:
        raise SystemExit("target repository does not contain the pinned Wolvrix commit")
    blob = _git_bytes(
        wolvrix_repo,
        ["show", f"{wolvrix_commit}:{PREFLIGHT_PROBE_SUBMODULE_PATH}"],
        "pinned preflight source blob",
    )
    if not blob:
        raise SystemExit("pinned preflight source blob is empty")
    return hashlib.sha256(
        challenge_nonce.encode("ascii") + b"\0" + blob
    ).hexdigest()


def _preflight_prompt(evaluator: ModuleType, challenge_nonce: str) -> str:
    """Build a small deterministic capability probe, not a research request."""
    smoke_hypothesis_json = json.dumps(
        PREFLIGHT_SMOKE_HYPOTHESIS, ensure_ascii=False
    )
    smoke_patch_json = json.dumps(PREFLIGHT_SMOKE_PATCH, ensure_ascii=False)
    return f"""\
CAPABILITY PREFLIGHT ONLY. This is not performance research, and this response
will never be evaluated or retained as an optimization. Do not inspect prior
experiments, generated code, profiles, or unrelated files. Do not modify the
checkout. Use a read-only repository shell, perform the pinned-blob challenge
below, and then return the JSON immediately.

You must invoke at least one repository tool. Read the tracked blob with
`git -C wolvrix show {evaluator.PINNED_WOLVRIX_COMMIT}:{PREFLIGHT_PROBE_SUBMODULE_PATH}`.
Compute lowercase SHA-256 over the exact byte concatenation
ASCII({challenge_nonce}) + one NUL byte + that blob. Hash the pinned blob, not
the worktree file, and do not use a previously published digest. The expected
digest is intentionally absent.

Return exactly one JSON object with these six fields and no Markdown or prose:
- `schema_version`: integer 2
- `candidate_mode`: string `default-path`
- `hypothesis`: exactly the JSON string {smoke_hypothesis_json}
- `evidence`: an array containing exactly one string,
  `repo_probe_attestation=<your 64 lowercase hex digest>`
- `patch`: exactly the JSON string {smoke_patch_json}
- `enable_options`: an empty array

Copy the supplied patch byte-for-byte, including its final newline. Do not
invent an optimization, return a placeholder, use control mode, or call more
tools after the digest has been computed.
"""


def _check_pinned_patch(
    target_repo: Path,
    evaluator: ModuleType,
    patch: str,
    changed_paths: tuple[str, ...],
) -> str | None:
    """Apply a smoke patch to isolated temporary Git index and object storage."""
    wolvrix_repo = target_repo / "wolvrix"
    try:
        with tempfile.TemporaryDirectory(prefix="simpletes-preflight-index-") as temp:
            temp_root = Path(temp)
            index_path = temp_root / "index"
            object_path = temp_root / "objects"
            object_path.mkdir(mode=0o700)
            common_dir_result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(wolvrix_repo),
                    "rev-parse",
                    "--path-format=absolute",
                    "--git-common-dir",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_clean_git_environment(),
                check=False,
            )
            if common_dir_result.returncode != 0:
                return "cannot locate pinned Wolvrix object storage"
            common_dir = Path(
                common_dir_result.stdout.decode("utf-8", errors="strict").strip()
            ).resolve(strict=True)
            alternate_objects = (common_dir / "objects").resolve(strict=True)
            if not alternate_objects.is_dir() or ":" in str(alternate_objects):
                return "cannot isolate pinned Wolvrix object storage"
            environment = _clean_git_environment(
                GIT_INDEX_FILE=str(index_path),
                GIT_OBJECT_DIRECTORY=str(object_path),
                GIT_ALTERNATE_OBJECT_DIRECTORIES=str(alternate_objects),
            )

            def run(arguments: list[str], *, input_bytes: bytes | None = None):
                return subprocess.run(
                    ["git", "-C", str(wolvrix_repo), *arguments],
                    input=input_bytes,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=environment,
                    check=False,
                )

            if run(["read-tree", evaluator.PINNED_WOLVRIX_COMMIT]).returncode != 0:
                return "cannot materialize the pinned Wolvrix tree for patch checking"
            for relative in changed_paths:
                listed = run(
                    [
                        "ls-tree",
                        "-z",
                        evaluator.PINNED_WOLVRIX_COMMIT,
                        "--",
                        relative,
                    ]
                )
                records = [item for item in listed.stdout.split(b"\0") if item]
                if listed.returncode != 0 or len(records) != 1:
                    return "capability patch does not target an existing pinned file"
                metadata, separator, encoded_path = records[0].partition(b"\t")
                fields = metadata.split()
                if (
                    separator != b"\t"
                    or encoded_path.decode("utf-8", errors="replace") != relative
                    or len(fields) != 3
                    or fields[0] not in {b"100644", b"100755"}
                    or fields[1] != b"blob"
                ):
                    return "capability patch target is not a regular pinned source file"
            applied = run(
                ["apply", "--cached", "--whitespace=error-all", "-"],
                input_bytes=patch.encode("utf-8"),
            )
            if applied.returncode != 0:
                return "capability patch does not apply cleanly to the pinned Wolvrix tree"
            difference = run(
                [
                    "diff-index",
                    "--cached",
                    "--quiet",
                    evaluator.PINNED_WOLVRIX_COMMIT,
                    "--",
                ]
            )
            if difference.returncode == 0:
                return "capability patch produces no pinned source change"
            if difference.returncode != 1:
                return "cannot verify the pinned source change from the capability patch"
    except (OSError, UnicodeError):
        return "cannot check the capability patch against the pinned Wolvrix tree"
    return None


def _validate_preflight_candidate(
    value: dict[str, Any],
    trace: Any,
    *,
    evaluator: ModuleType,
    expected_digest: str,
    target_repo: Path | None = None,
) -> str | None:
    """Return a repair-safe reason when a capability response is not usable."""
    try:
        candidate = evaluator.parse_candidate_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True)
        )
    except evaluator.CandidateError:
        return "response is not a valid production GrhSIM candidate"
    if candidate.is_control or candidate.candidate_mode == "control":
        return "capability response must not use the control candidate mode"
    if _PLACEHOLDER_RE.fullmatch(candidate.hypothesis):
        return "capability response hypothesis is placeholder text"
    if candidate.hypothesis != PREFLIGHT_SMOKE_HYPOTHESIS:
        return "capability response did not copy the smoke hypothesis exactly"
    if any(_PLACEHOLDER_RE.fullmatch(item.strip()) for item in candidate.evidence):
        return "capability response evidence contains placeholder text"
    if candidate.patch != PREFLIGHT_SMOKE_PATCH:
        return "capability response did not copy the deterministic smoke patch exactly"
    try:
        changed_paths = evaluator.validate_patch(candidate.patch)
    except evaluator.CandidateError:
        return "capability response does not contain a valid safe unified diff"
    if not changed_paths:
        return "capability response patch changes no source files"
    if target_repo is not None:
        patch_error = _check_pinned_patch(
            target_repo,
            evaluator,
            candidate.patch,
            changed_paths,
        )
        if patch_error is not None:
            return patch_error

    reported = [
        item.removeprefix(PREFLIGHT_EVIDENCE_PREFIX)
        for item in candidate.evidence
        if item.startswith(PREFLIGHT_EVIDENCE_PREFIX)
    ]
    if (
        len(candidate.evidence) != 1
        or len(reported) != 1
        or re.fullmatch(r"[0-9a-f]{64}", reported[0]) is None
    ):
        return "capability response must report exactly one lowercase repo attestation"
    if reported[0] != expected_digest:
        return "reported repo attestation does not match the fresh pinned-blob challenge"
    if getattr(trace, "repo_tool_call_count", 0) < 1:
        return "capability response did not use a repository inspection tool"
    return None


def run_codex_preflight(args: argparse.Namespace) -> int:
    """Run a bounded repository-grounded capability smoke without research."""
    model = _selected_model(args)
    reasoning_effort = _selected_reasoning_effort(args)
    codex_exec_retries = _selected_codex_exec_retries(args)
    codex_capacity_continuations = _selected_codex_capacity_continuations(args)
    codex_transient_continuations = _selected_codex_transient_continuations(
        args
    )
    is_k3 = model == "k3"
    output_mode = "local-json" if is_k3 else "provider-structured"
    tool_choice_mode = "required-first" if is_k3 else "auto"
    target_repo = args.target_repo.expanduser().resolve()
    if not (target_repo / ".git").exists() or not (target_repo / "env.sh").is_file():
        raise SystemExit(f"not a GrhSIM playground checkout: {target_repo}")
    codex_config = _regular_file(args.codex_config, "Codex config")
    codex_auth = _regular_file(args.codex_auth, "Codex auth")
    schema = _regular_file(TASK_ROOT / "candidate.schema.json", "candidate schema")
    local_schema = _regular_file(
        LOCAL_VALIDATION_SCHEMA, "candidate local validation schema"
    )
    k3_model_catalog = (
        _regular_file(K3_MODEL_CATALOG, "K3 model catalog")
        if is_k3
        else None
    )
    max_agent_threads = getattr(
        args,
        "codex_max_agent_threads",
        DEFAULT_CODEX_MAX_AGENT_THREADS,
    )
    if is_k3 and not 1 <= max_agent_threads <= 32:
        raise SystemExit("--codex-max-agent-threads must be in 1..32")
    evaluator = _load_evaluator()
    preflight_timeout = getattr(
        args, "preflight_timeout", DEFAULT_PREFLIGHT_TIMEOUT
    )
    if (
        isinstance(preflight_timeout, bool)
        or not isinstance(preflight_timeout, (int, float))
        or not 0 < preflight_timeout <= 1_800
    ):
        raise SystemExit("preflight timeout must be in (0, 1800] seconds")
    challenge_nonce = secrets.token_hex(32)
    expected_digest = _pinned_probe_digest(
        target_repo, evaluator, challenge_nonce
    )

    from simpletes.llm.codex_exec import CodexExecClient
    from simpletes.llm.types import LLMCallError

    try:
        client = CodexExecClient(
            model=model,
            reasoning_effort=reasoning_effort,
            config_path=str(codex_config),
            auth_path=str(codex_auth),
            repo_root=str(target_repo),
            output_schema=str(schema),
            local_validation_schema=str(local_schema),
            output_mode=output_mode,
            tool_choice_mode=tool_choice_mode,
            timeout=float(preflight_timeout),
            max_repair_attempts=0,
            max_agent_threads=(max_agent_threads if is_k3 else None),
            model_catalog_path=(
                str(k3_model_catalog) if k3_model_catalog is not None else None
            ),
            attempt_artifact_dir=str(PREFLIGHT_ATTEMPT_ARTIFACT_ROOT),
            runtime_home_dir=str(PREFLIGHT_RUNTIME_HOME_ROOT),
            max_exec_retries=codex_exec_retries,
            max_capacity_continuations=codex_capacity_continuations,
            max_transient_continuations=codex_transient_continuations,
        )
    except ValueError as error:
        raise SystemExit(
            f"invalid Codex capability preflight configuration: {error}"
        ) from None
    try:
        prompt = _preflight_prompt(evaluator, challenge_nonce)

        def validate(value: dict[str, Any], trace: Any) -> str | None:
            return _validate_preflight_candidate(
                value,
                trace,
                evaluator=evaluator,
                expected_digest=expected_digest,
                target_repo=target_repo,
            )

        try:
            result = asyncio.run(client.capability_probe(prompt, validate=validate))
        except LLMCallError as error:
            # The backend message is already bounded and scrubbed against the
            # loaded auth material and credential-shaped diagnostics.  Keep it
            # visible so a failed capability gate can be repaired without a
            # blind second provider call; suppress the exception chain so raw
            # subprocess/provider payloads still cannot reach a traceback.
            raise SystemExit(
                "Codex capability preflight failed "
                f"({error.error_type}): {error.message}"
            ) from None
        if is_k3 and any(
            "Model metadata for k3 not found" in diagnostic
            for diagnostic in getattr(result.trace, "diagnostic_summaries", ())
        ):
            raise SystemExit(
                "Codex capability preflight used fallback K3 model metadata"
            )
    finally:
        client.close()
    print(
        "Codex capability preflight passed: "
        f"model={model}, effort={reasoning_effort}, "
        f"repo_tool_calls={result.trace.repo_tool_call_count}, "
        f"model_catalog={'validated' if is_k3 else 'native'}, "
        f"response_chars={len(result.canonical)}, "
        "response_sha256="
        f"{hashlib.sha256(result.canonical.encode('utf-8')).hexdigest()}"
    )
    return 0


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    model = _selected_model(args)
    reasoning_effort = _selected_reasoning_effort(args)
    codex_exec_retries = _selected_codex_exec_retries(args)
    codex_capacity_continuations = _selected_codex_capacity_continuations(args)
    codex_transient_continuations = _selected_codex_transient_continuations(
        args
    )
    is_k3 = model == "k3"
    output_mode = "local-json" if is_k3 else "provider-structured"
    tool_choice_mode = "required-first" if is_k3 else "auto"
    target_repo = args.target_repo.expanduser().resolve()
    if not (target_repo / ".git").exists() or not (target_repo / "env.sh").is_file():
        raise SystemExit(f"not a GrhSIM playground checkout: {target_repo}")
    target_env = _regular_file(target_repo / "env.sh", "target env.sh")
    codex_config = _regular_file(args.codex_config, "Codex config")
    codex_auth = _regular_file(args.codex_auth, "Codex auth")
    schema = _regular_file(TASK_ROOT / "candidate.schema.json", "candidate schema")
    local_schema = _regular_file(
        LOCAL_VALIDATION_SCHEMA, "candidate local validation schema"
    )
    k3_model_catalog = (
        _regular_file(K3_MODEL_CATALOG, "K3 model catalog")
        if is_k3
        else None
    )
    init_program = _regular_file(args.init_program, "initial program")
    evaluator = _regular_file(TASK_ROOT / "evaluator.py", "evaluator")
    instruction = _regular_file(TASK_ROOT / "instruction.txt", "instruction")
    k3_instruction_suffix = (
        _regular_file(K3_INSTRUCTION_SUFFIX, "K3 instruction suffix")
        if is_k3
        else None
    )
    evaluator_contract = _load_evaluator()
    init_candidate = _parse_seed(init_program, evaluator_contract, "initial program")
    if init_program.name.startswith("best_program.") and init_program.parent.name.startswith(
        "db_state_"
    ):
        validated_init = _validate_checkpoint_contract(
            init_program.parent,
            evaluator_contract,
            expected_candidate=init_candidate,
        )
        init_program = validated_init.seed_path
    resume_state: Path | None = None
    if args.resume is not None:
        selected_state = _checkpoint_state_dir(args.resume, "resume checkpoint")
        validated_resume = _validate_checkpoint_contract(
            selected_state, evaluator_contract
        )
        # Pass the exact state and seed which were validated. Passing the
        # instance directory would make SimpleTES select "latest" a second
        # time, allowing a concurrently-published checkpoint to bypass these
        # contract checks.
        resume_state = validated_resume.state_dir
        init_program = validated_resume.seed_path

    extend_resume_budget = bool(
        getattr(args, "extend_resume_budget", False)
    )
    if extend_resume_budget and resume_state is None:
        raise SystemExit("--extend-resume-budget requires --resume")
    # An already-extended checkpoint must be restartable at its exact persisted
    # absolute limit without asking the engine to extend it a second time.  The
    # engine still fail-closes any non-exact resume unless the explicit
    # extension flag is present, so allowing the wider launcher range for every
    # resume does not make implicit budget growth possible.
    proposal_limit = (
        MAX_EXTENDED_PROPOSALS if resume_state is not None else MAX_PROPOSALS
    )
    if not 1 <= args.max_proposals <= proposal_limit:
        raise SystemExit(
            f"--max-proposals must be in 1..{proposal_limit}"
        )
    if not 1 <= args.valid_target <= args.max_proposals:
        raise SystemExit("--valid-target must be in 1..max-proposals")
    gen_concurrency = getattr(
        args, "gen_concurrency", DEFAULT_GEN_CONCURRENCY
    )
    if not 1 <= gen_concurrency <= NUM_CHAINS:
        raise SystemExit(
            f"--gen-concurrency must be in 1..{NUM_CHAINS}"
        )
    max_agent_threads = getattr(
        args,
        "codex_max_agent_threads",
        DEFAULT_CODEX_MAX_AGENT_THREADS,
    )
    if is_k3 and not 1 <= max_agent_threads <= 32:
        raise SystemExit("--codex-max-agent-threads must be in 1..32")

    output_path = args.output_path
    if output_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = SIMPLETES_ROOT / "checkpoints" / "grhsim_simtop_50k" / stamp
    output_path = output_path.expanduser().resolve()

    command = [
        sys.executable,
        str(SIMPLETES_ROOT / "main.py"),
        "--init-program",
        str(init_program),
        "--evaluator",
        str(evaluator),
        "--instruction",
        str(instruction),
        "--llm-backend",
        "codex_exec",
        "--codex-config",
        str(codex_config),
        "--codex-auth",
        str(codex_auth),
        "--codex-repo-root",
        str(target_repo),
        "--codex-output-schema",
        str(schema),
        "--codex-local-validation-schema",
        str(local_schema),
        "--codex-output-mode",
        output_mode,
        "--codex-tool-choice-mode",
        tool_choice_mode,
        "--model",
        model,
        "--reasoning-effort",
        reasoning_effort,
        "--selector",
        "rpucg",
        "--num-chains",
        str(NUM_CHAINS),
        "--k-candidates",
        "1",
        "--eval-concurrency",
        "1",
        "--gen-concurrency",
        str(gen_concurrency),
        "--init-eval-repeats",
        "1",
        "--max-generations",
        str(args.max_proposals),
        "--max-valid-evaluations",
        str(args.valid_target),
        "--eval-timeout",
        str(args.eval_timeout),
        "--timeout",
        str(args.llm_timeout),
        "--retry",
        str(codex_exec_retries),
        "--codex-capacity-continuations",
        str(codex_capacity_continuations),
        "--codex-transient-continuations",
        str(codex_transient_continuations),
        "--max-tokens",
        str(args.max_tokens),
        "--disable-reflection",
        # This launcher performs the stronger repository-grounded capability
        # gate immediately before spawning main.py.  Do not repeat the generic
        # backend preflight after an engine process has been created.
        "--skip-preflight",
        "--log-interval",
        "1",
        "--db-show-interval",
        "1",
        "--save-llm-io",
        "--output-path",
        str(output_path),
    ]
    if is_k3:
        assert k3_model_catalog is not None
        assert k3_instruction_suffix is not None
        command.extend(
            [
                "--instruction-suffix",
                str(k3_instruction_suffix),
                "--codex-max-agent-threads",
                str(max_agent_threads),
                "--codex-model-catalog",
                str(k3_model_catalog),
            ]
        )
    if resume_state is not None:
        command.extend(["--resume", str(resume_state)])
    if extend_resume_budget:
        command.append("--extend-resume-budget")
    command = [
        "bash",
        "-c",
        'set -euo pipefail\nsource "$1" >/dev/null\nshift\nexec "$@"',
        "grhsim-simpletes-launcher",
        str(target_env),
        *command,
    ]

    environment = {
        name: value
        for name, value in os.environ.items()
        if not any(
            fragment in name.upper()
            for fragment in (
                "API_KEY",
                "APIKEY",
                "TOKEN",
                "PASSWORD",
                "SECRET",
                "CREDENTIAL",
                "WEBHOOK",
                "COOKIE",
                "PRIVATE_KEY",
            )
        )
    }
    environment["GRHSIM_SOURCE_REPO"] = str(target_repo)
    environment["GRHSIM_VALID_CANDIDATE_TARGET"] = str(args.valid_target)
    # Formal runs never inherit shortcuts that disable correctness/attribution
    # gates from an interactive parent shell.
    environment["GRHSIM_VERIFY_DEFAULT_OFF"] = "1"
    environment["GRHSIM_RUN_FUNCTION_GATES"] = "1"
    environment["GRHSIM_RUN_FOCUSED_TESTS"] = "1"
    if args.slot_root is not None:
        environment["GRHSIM_SLOT_ROOT"] = str(args.slot_root.expanduser().resolve())
    return command, environment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-repo", type=Path, default=DEFAULT_TARGET_REPO)
    parser.add_argument("--codex-config", type=Path, default=DEFAULT_CODEX_CONFIG)
    parser.add_argument("--codex-auth", type=Path, default=DEFAULT_CODEX_AUTH)
    parser.add_argument(
        "--model",
        default=MODEL,
        help=f"Codex model for this run (default: {MODEL})",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=REASONING_EFFORT,
        help=f"Codex reasoning effort for this run (default: {REASONING_EFFORT})",
    )
    parser.add_argument(
        "--init-program",
        type=Path,
        default=DEFAULT_INIT_PROGRAM,
        help="Initial marked candidate; use a prior best_program.txt for a fresh continuation",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--slot-root", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--extend-resume-budget",
        action="store_true",
        help=(
            "Monotonically extend an exact resume checkpoint's absolute "
            "proposal/valid limits while preserving its accumulated state"
        ),
    )
    parser.add_argument("--max-proposals", type=int, default=16)
    parser.add_argument("--valid-target", type=int, default=8)
    parser.add_argument("--eval-timeout", type=float, default=21_600.0)
    parser.add_argument(
        "--llm-timeout",
        type=float,
        default=DEFAULT_LLM_TIMEOUT,
        help=(
            "Per-generation Codex timeout in seconds "
            f"(default: {DEFAULT_LLM_TIMEOUT:g})"
        ),
    )
    parser.add_argument(
        "--gen-concurrency",
        type=int,
        default=DEFAULT_GEN_CONCURRENCY,
        help=(
            "Concurrent Codex generation workers; bounded by the "
            f"{NUM_CHAINS} research chains "
            f"(default: {DEFAULT_GEN_CONCURRENCY})"
        ),
    )
    parser.add_argument(
        "--codex-max-agent-threads",
        type=int,
        default=DEFAULT_CODEX_MAX_AGENT_THREADS,
        help=(
            "Maximum concurrently open K3-spawned subagents; this is not "
            "passed to other models "
            f"(default: {DEFAULT_CODEX_MAX_AGENT_THREADS})"
        ),
    )
    parser.add_argument(
        "--codex-exec-retries",
        type=int,
        default=DEFAULT_CODEX_EXEC_RETRIES,
        help=(
            "Retries for diagnosed transient non-zero Codex exits "
            f"(default: {DEFAULT_CODEX_EXEC_RETRIES})"
        ),
    )
    parser.add_argument(
        "--codex-capacity-continuations",
        type=int,
        default=DEFAULT_CODEX_CAPACITY_CONTINUATIONS,
        help=(
            "In-session Codex 'continue' turns for model-at-capacity before "
            "using a normal exec retry "
            f"(default: {DEFAULT_CODEX_CAPACITY_CONTINUATIONS})"
        ),
    )
    parser.add_argument(
        "--codex-transient-continuations",
        type=int,
        default=DEFAULT_CODEX_TRANSIENT_CONTINUATIONS,
        help=(
            "In-session Codex 'continue' turns for remote-compaction and "
            "reconnect/stream failures before using a normal exec retry "
            f"(default: {DEFAULT_CODEX_TRANSIENT_CONTINUATIONS})"
        ),
    )
    parser.add_argument(
        "--preflight-timeout",
        type=float,
        default=DEFAULT_PREFLIGHT_TIMEOUT,
        help=(
            "Per-launch deterministic capability-gate timeout; independent "
            "from the longer research-generation timeout"
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=32_768)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate model/provider/auth/schema and exit without creating a research run",
    )
    args = parser.parse_args()

    if args.preflight_only:
        return run_codex_preflight(args)
    command, environment = build_command(args)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    launch_guard = _launch_guard_digest(args, command)
    run_codex_preflight(args)
    if not secrets.compare_digest(
        launch_guard, _launch_guard_digest(args, command)
    ):
        raise SystemExit(
            "SimpleTES launch inputs changed during capability preflight; "
            "refusing to spawn research"
        )
    completed = subprocess.run(command, cwd=SIMPLETES_ROOT, env=environment, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
