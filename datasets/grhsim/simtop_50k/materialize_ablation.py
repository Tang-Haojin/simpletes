#!/usr/bin/env python3
"""Materialize exact or mechanically composed checkpoint ablation candidates."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from evaluator import parse_candidate_text, validate_patch


START_MARKER = "# EVOLVE-BLOCK-START"
END_MARKER = "# EVOLVE-BLOCK-END"
SOURCE_PATH = "lib/emit/grhsim_cpp.cpp"
HISTORICAL_RWA_BASELINE_WOLVRIX_COMMIT = (
    "8f6ba14397b0c3d00cb909153af1c6464f4f1ed9"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nodes_json", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--gen-id", type=int)
    selection.add_argument(
        "--compose-rwa",
        nargs=3,
        type=int,
        metavar=("RW_GEN", "RWF_GEN", "RWFA_GEN"),
        help=(
            "compose RWA as RW plus the RWF-to-RWFA delta, then prove it is "
            "byte-identical to RWFA with the RW-to-RWF delta removed"
        ),
    )
    parser.add_argument(
        "--wolvrix-repo",
        type=Path,
        help="Wolvrix repository containing the historical RWA baseline object",
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _find_node(nodes: list[dict[str, object]], gen_id: int) -> dict[str, object]:
    matches = [node for node in nodes if node.get("gen_id") == gen_id]
    if len(matches) != 1:
        raise ValueError(f"expected one node for gen {gen_id}, found {len(matches)}")
    return matches[0]


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _document(payload: dict[str, object]) -> str:
    return (
        f"{START_MARKER}\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        f"{END_MARKER}\n"
    )


def _canonical_diff(before: str, after: str) -> str:
    if before == after:
        return ""
    body = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{SOURCE_PATH}",
            tofile=f"b/{SOURCE_PATH}",
            n=3,
        )
    )
    return f"diff --git a/{SOURCE_PATH} b/{SOURCE_PATH}\n{body}"


def _apply_patch(source: str, patch: str, *, reverse: bool = False) -> str:
    validate_patch(patch)
    with tempfile.TemporaryDirectory(prefix="simpletes-rwa-") as temp_dir:
        root = Path(temp_dir)
        target = root / SOURCE_PATH
        target.parent.mkdir(parents=True)
        target.write_text(source, encoding="utf-8")
        command = ["git", "apply", "--whitespace=error-all"]
        if reverse:
            command.append("--reverse")
        completed = subprocess.run(
            command,
            cwd=root,
            input=patch,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ValueError(f"git apply failed: {detail}")
        return target.read_text(encoding="utf-8")


def _checkpoint_candidate(
    nodes: list[dict[str, object]], gen_id: int
):
    node = _find_node(nodes, gen_id)
    code = node.get("code")
    if not isinstance(code, str):
        raise ValueError(f"node at gen {gen_id} has no candidate text")
    candidate = parse_candidate_text(code)
    if candidate.candidate_mode != "default-path" or candidate.enable_options:
        raise ValueError(
            f"gen {gen_id} must be a zero-option default-path candidate"
        )
    files = validate_patch(candidate.patch)
    if files != (SOURCE_PATH,):
        raise ValueError(
            f"gen {gen_id} must modify only {SOURCE_PATH}, got {files!r}"
        )
    return node, candidate


def _compose_rwa(
    nodes: list[dict[str, object]],
    *,
    rw_gen: int,
    rwf_gen: int,
    rwfa_gen: int,
    baseline: str,
    baseline_wolvrix_commit: str,
    label: str,
) -> tuple[str, dict[str, object]]:
    rw_node, rw_candidate = _checkpoint_candidate(nodes, rw_gen)
    rwf_node, rwf_candidate = _checkpoint_candidate(nodes, rwf_gen)
    rwfa_node, rwfa_candidate = _checkpoint_candidate(nodes, rwfa_gen)

    rw_source = _apply_patch(baseline, rw_candidate.patch)
    rwf_source = _apply_patch(baseline, rwf_candidate.patch)
    rwfa_source = _apply_patch(baseline, rwfa_candidate.patch)
    a_patch = _canonical_diff(rwf_source, rwfa_source)
    f_patch = _canonical_diff(rw_source, rwf_source)
    validate_patch(a_patch)
    validate_patch(f_patch)

    rwa_via_add = _apply_patch(rw_source, a_patch)
    rwa_via_remove = _apply_patch(rwfa_source, f_patch, reverse=True)
    if rwa_via_add != rwa_via_remove:
        raise ValueError("RWA composition paths RW+A and RWFA-F are not byte-identical")
    if _apply_patch(rwa_via_add, f_patch) != rwfa_source:
        raise ValueError("RWA+F does not close byte-identically to RWFA")
    if _apply_patch(rwf_source, a_patch) != rwfa_source:
        raise ValueError("RWF+A does not reproduce RWFA")

    rwa_patch = _canonical_diff(baseline, rwa_via_add)
    validate_patch(rwa_patch)
    source_rows = []
    for gen_id, node, candidate in (
        (rw_gen, rw_node, rw_candidate),
        (rwf_gen, rwf_node, rwf_candidate),
        (rwfa_gen, rwfa_node, rwfa_candidate),
    ):
        source_rows.append(
            {
                "gen_id": gen_id,
                "node_id": node["id"],
                "candidate_digest": candidate.digest,
                "patch_sha256": _sha256_text(candidate.patch),
            }
        )
    composition = {
        "baseline_source_sha256": _sha256_text(baseline),
        "a_patch_sha256": _sha256_text(a_patch),
        "f_patch_sha256": _sha256_text(f_patch),
        "rwa_patch_sha256": _sha256_text(rwa_patch),
        "rwa_source_sha256": _sha256_text(rwa_via_add),
        "rwa_source_lines": len(rwa_via_add.splitlines()),
        "rw_plus_a_equals_rwfa_minus_f": True,
        "rwa_plus_f_equals_rwfa": True,
        "rwf_plus_a_equals_rwfa": True,
    }
    payload = {
        "schema_version": 2,
        "candidate_mode": "default-path",
        "hypothesis": (
            f"Fresh controlled ablation arm {label}: mechanically retain RW and A "
            "while removing F, to measure B-to-RWA end-to-end performance without "
            "assuming additive gains."
        ),
        "evidence": [
            (
                "Mechanically composed from checkpoint gens "
                f"{rw_gen}/{rwf_gen}/{rwfa_gen}; input node IDs are "
                f"{rw_node['id']}/{rwf_node['id']}/{rwfa_node['id']}."
            ),
            (
                "Derived A as RWF-to-RWFA and F as RW-to-RWF; "
                f"A patch SHA-256 {_sha256_text(a_patch)}, F patch SHA-256 "
                f"{_sha256_text(f_patch)}."
            ),
            (
                "Independent RW+A and RWFA-F construction paths are byte-identical; "
                "RWA+F and RWF+A both reproduce RWFA byte-for-byte."
            ),
            (
                f"Full pinned-baseline-to-RWA patch SHA-256 {_sha256_text(rwa_patch)}; "
                f"materialized source SHA-256 {_sha256_text(rwa_via_add)}."
            ),
        ],
        "patch": rwa_patch,
        "enable_options": [],
    }
    document = _document(payload)
    candidate = parse_candidate_text(document)
    report = {
        "digest": candidate.digest,
        "label": label,
        "mode": "compose-rwa",
        "baseline_wolvrix_commit": baseline_wolvrix_commit,
        "source_nodes": source_rows,
        "composition": composition,
    }
    return document, report


def _historical_rwa_source(wolvrix_repo: Path) -> str:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(wolvrix_repo),
            "show",
            f"{HISTORICAL_RWA_BASELINE_WOLVRIX_COMMIT}:{SOURCE_PATH}",
        ],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"cannot read historical RWA Wolvrix source: {detail}")
    return completed.stdout.decode("utf-8")


def _materialize_exact(
    nodes: list[dict[str, object]], *, gen_id: int, label: str
) -> tuple[str, dict[str, object]]:
    node = _find_node(nodes, gen_id)
    code = node.get("code")
    if not isinstance(code, str):
        raise ValueError(f"node at gen {gen_id} has no candidate text")
    source = parse_candidate_text(code)

    payload = {
        "schema_version": source.schema_version,
        "candidate_mode": source.candidate_mode,
        "hypothesis": (
            f"Fresh controlled ablation arm {label}, mechanically preserving "
            f"the exact source patch and enable options from checkpoint gen "
            f"{gen_id}. Original hypothesis: {source.hypothesis}"
        ),
        "evidence": [
            *source.evidence,
            (
                f"Materialized from node {node['id']} at gen {gen_id}; "
                "only hypothesis/evidence metadata changed to give this fresh "
                "replication a distinct evaluator digest."
            ),
        ],
        "patch": source.patch,
        "enable_options": [
            {"name": name, "value": value}
            for name, value in source.enable_options.items()
        ],
    }
    document = _document(payload)
    candidate = parse_candidate_text(document)
    report = {
        "digest": candidate.digest,
        "gen_id": gen_id,
        "label": label,
        "node_id": node["id"],
    }
    return document, report


def main() -> int:
    args = _parse_args()
    nodes = json.loads(args.nodes_json.read_text(encoding="utf-8"))
    if not isinstance(nodes, list):
        raise SystemExit("nodes JSON must be an array")
    try:
        if args.compose_rwa is not None:
            if args.wolvrix_repo is None:
                raise ValueError("--wolvrix-repo is required with --compose-rwa")
            document, report = _compose_rwa(
                nodes,
                rw_gen=args.compose_rwa[0],
                rwf_gen=args.compose_rwa[1],
                rwfa_gen=args.compose_rwa[2],
                baseline=_historical_rwa_source(args.wolvrix_repo),
                baseline_wolvrix_commit=HISTORICAL_RWA_BASELINE_WOLVRIX_COMMIT,
                label=args.label,
            )
        else:
            document, report = _materialize_exact(
                nodes, gen_id=args.gen_id, label=args.label
            )
    except (ValueError, subprocess.SubprocessError) as exc:
        raise SystemExit(str(exc)) from None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    report["output"] = str(args.output)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
