#!/usr/bin/env python3
"""Launch the bounded GrhSIM SimTop-50k SimpleTES research run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import ModuleType
from typing import Any


TASK_ROOT = Path(__file__).resolve().parent
SIMPLETES_ROOT = TASK_ROOT.parents[2]
DEFAULT_TARGET_REPO = SIMPLETES_ROOT.parent / "wolvrix-playground-gsim-calibrate-5"
DEFAULT_CODEX_CONFIG = Path("~/.codex/config.mjy.toml").expanduser()
DEFAULT_CODEX_AUTH = Path("~/.codex/auth.mjy.json").expanduser()
DEFAULT_INIT_PROGRAM = TASK_ROOT / "init_program.txt"
MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "ultra"
MAX_PROPOSALS = 64


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


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    target_repo = args.target_repo.expanduser().resolve()
    if not (target_repo / ".git").exists() or not (target_repo / "env.sh").is_file():
        raise SystemExit(f"not a GrhSIM playground checkout: {target_repo}")
    target_env = _regular_file(target_repo / "env.sh", "target env.sh")
    codex_config = _regular_file(args.codex_config, "Codex config")
    codex_auth = _regular_file(args.codex_auth, "Codex auth")
    schema = _regular_file(TASK_ROOT / "candidate.schema.json", "candidate schema")
    init_program = _regular_file(args.init_program, "initial program")
    evaluator = _regular_file(TASK_ROOT / "evaluator.py", "evaluator")
    instruction = _regular_file(TASK_ROOT / "instruction.txt", "instruction")
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

    if not 1 <= args.max_proposals <= MAX_PROPOSALS:
        raise SystemExit(f"--max-proposals must be in 1..{MAX_PROPOSALS}")
    if not 1 <= args.valid_target <= args.max_proposals:
        raise SystemExit("--valid-target must be in 1..max-proposals")

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
        "--model",
        MODEL,
        "--reasoning-effort",
        REASONING_EFFORT,
        "--selector",
        "rpucg",
        "--num-chains",
        "4",
        "--k-candidates",
        "1",
        "--eval-concurrency",
        "1",
        "--gen-concurrency",
        "1",
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
        "--max-tokens",
        str(args.max_tokens),
        "--disable-reflection",
        "--log-interval",
        "1",
        "--db-show-interval",
        "1",
        "--save-llm-io",
        "--output-path",
        str(output_path),
    ]
    if resume_state is not None:
        command.extend(["--resume", str(resume_state)])
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
        "--init-program",
        type=Path,
        default=DEFAULT_INIT_PROGRAM,
        help="Initial marked candidate; use a prior best_program.txt for a fresh continuation",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--slot-root", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-proposals", type=int, default=16)
    parser.add_argument("--valid-target", type=int, default=8)
    parser.add_argument("--eval-timeout", type=float, default=21_600.0)
    parser.add_argument("--llm-timeout", type=float, default=3_000.0)
    parser.add_argument("--max-tokens", type=int, default=32_768)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    command, environment = build_command(args)
    if args.dry_run:
        print(shlex.join(command))
        return 0
    completed = subprocess.run(command, cwd=SIMPLETES_ROOT, env=environment, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
