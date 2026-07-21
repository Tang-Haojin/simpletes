#!/usr/bin/env python3
"""Launch the bounded GrhSIM SimTop-50k SimpleTES research run."""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys


TASK_ROOT = Path(__file__).resolve().parent
SIMPLETES_ROOT = TASK_ROOT.parents[2]
DEFAULT_TARGET_REPO = SIMPLETES_ROOT.parent / "wolvrix-playground-gsim-calibrate-5"
DEFAULT_CODEX_CONFIG = Path("~/.codex/config.mjy.toml").expanduser()
DEFAULT_CODEX_AUTH = Path("~/.codex/auth.mjy.json").expanduser()
DEFAULT_INIT_PROGRAM = TASK_ROOT / "init_program.txt"
MODEL = "gpt-5.6-sol"
REASONING_EFFORT = "ultra"
MAX_PROPOSALS = 64


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
    if args.resume is not None:
        command.extend(["--resume", str(args.resume.expanduser().resolve())])
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
    # Formal runs never inherit shortcuts that disable correctness/default-off
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
