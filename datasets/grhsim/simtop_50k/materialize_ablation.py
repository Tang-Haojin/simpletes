#!/usr/bin/env python3
"""Materialize a checkpoint node as a uniquely labeled ablation candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluator import parse_candidate_text


START_MARKER = "# EVOLVE-BLOCK-START"
END_MARKER = "# EVOLVE-BLOCK-END"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nodes_json", type=Path)
    parser.add_argument("--gen-id", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    nodes = json.loads(args.nodes_json.read_text(encoding="utf-8"))
    matches = [node for node in nodes if node.get("gen_id") == args.gen_id]
    if len(matches) != 1:
        raise SystemExit(
            f"expected one node for gen {args.gen_id}, found {len(matches)}"
        )

    source = parse_candidate_text(matches[0]["code"])
    payload = {
        "schema_version": source.schema_version,
        "hypothesis": (
            f"Fresh controlled ablation arm {args.label}, mechanically preserving "
            f"the exact source patch and enable options from checkpoint gen "
            f"{args.gen_id}. Original hypothesis: {source.hypothesis}"
        ),
        "evidence": [
            *source.evidence,
            (
                f"Materialized from node {matches[0]['id']} at gen {args.gen_id}; "
                "only hypothesis/evidence metadata changed to give this fresh "
                "replication a distinct evaluator digest."
            ),
        ],
        "patch": source.patch,
        "enable_options": source.enable_options,
    }
    document = (
        f"{START_MARKER}\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        f"{END_MARKER}\n"
    )
    candidate = parse_candidate_text(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(
        json.dumps(
            {
                "digest": candidate.digest,
                "gen_id": args.gen_id,
                "label": args.label,
                "node_id": matches[0]["id"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
