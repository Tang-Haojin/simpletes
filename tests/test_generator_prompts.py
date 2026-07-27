"""Regression tests for backend-specific generation prompt contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from simpletes.config import EngineConfig
from simpletes.generator import Generator
from simpletes.node import Node
from simpletes.utils.code_extract import EvolveBlockContext


def _generator(*, backend: str, selector: str) -> Generator:
    generator = object.__new__(Generator)
    generator._config = EngineConfig(
        init_program="candidate.txt",
        evaluator_path="evaluator.py",
        instruction_path="instruction.txt",
        llm_backend=backend,
        selector=selector,
    )
    generator._instruction = (
        "FULL TASK: inspect the repository and return a measured optimization.\n"
        "Keep every task-specific constraint in this prompt."
    )
    generator._evolve_context = EvolveBlockContext(
        prefix="# fixed prefix\n# EVOLVE-BLOCK-START",
        suffix="# EVOLVE-BLOCK-END\n# fixed suffix",
        has_markers=True,
    )
    generator._available_packages = ["jsonschema==4"]
    generator._code_fence_tag = "text"
    generator._language_name = "text"
    return generator


def _inspiration() -> Node:
    return Node(
        id="inspiration-1",
        code='{"hypothesis": "FULL INSPIRATION PAYLOAD"}',
        score=1.25,
        metrics={"combined_score": 1.25, "walltime_ms": 53791.5},
        reflection="Retain the complete inspiration and its evidence.",
    )


@pytest.mark.parametrize("selector", ["balance", "rpucg"])
def test_codex_exec_prompt_has_one_unambiguous_structured_response_contract(
    selector: str,
) -> None:
    generator = _generator(backend="codex_exec", selector=selector)

    prompt = generator.build_prompt(
        [_inspiration()],
        failure_patterns={"invalid candidate": 0.25},
        policy_context="FULL POLICY CONTEXT",
        shared_construction_summary="FULL SHARED CONSTRUCTION",
    )

    assert "Return exactly one JSON object" in prompt
    assert "single contract-conforming JSON object" in prompt
    assert "response contract described by the task" in prompt
    assert "no Markdown code fence, no EVOLVE-BLOCK marker" in prompt
    assert "backend validates the object and adds the extraction markers" in prompt

    # These legacy instructions contradict codex exec's output-schema contract.
    assert "Only the code between" not in prompt
    assert "Keep marker lines exactly as written" not in prompt
    assert "Return one text code block" not in prompt
    assert "EXACT_PREFIX" not in prompt
    assert "EXACT_SUFFIX" not in prompt

    # Prompt specialization must not trim the actual research task or context.
    assert generator._instruction in prompt
    assert "FULL INSPIRATION PAYLOAD" in prompt
    assert "walltime_ms: 53791.500000" in prompt
    assert "FULL POLICY CONTEXT" in prompt
    assert "FULL SHARED CONSTRUCTION" in prompt
    assert "invalid candidate: 25.0%" in prompt
    assert "jsonschema==4" in prompt


def test_grhsim_codex_exec_real_prompt_preserves_full_task_and_inspiration() -> None:
    dataset = Path(__file__).parents[1] / "datasets" / "grhsim" / "simtop_50k"
    instruction = (dataset / "instruction.txt").read_text(encoding="utf-8")
    init_program = (dataset / "init_program.txt").read_text(encoding="utf-8")
    generator = _generator(backend="codex_exec", selector="rpucg")
    generator._instruction = instruction
    generator._evolve_context = EvolveBlockContext.from_program(init_program)
    inspiration = Node(
        id="post-rwa-control",
        code=init_program,
        score=0.0,
        metrics={"combined_score": 0.0, "walltime_ms": 53794.5},
    )

    prompt = generator.build_prompt([inspiration])

    assert instruction in prompt
    assert init_program in prompt
    assert "Return exactly one JSON object" in prompt
    assert "Only the code between" not in prompt
    assert "Keep marker lines exactly as written" not in prompt
    assert "Return one text code block" not in prompt
    assert "EXACT_PREFIX" not in prompt
    assert "EXACT_SUFFIX" not in prompt


@pytest.mark.parametrize("selector", ["balance", "rpucg"])
@pytest.mark.parametrize("backend", ["litellm", "vllm_token_forcing"])
def test_non_structured_backends_keep_legacy_marker_and_fence_contract(
    backend: str,
    selector: str,
) -> None:
    prompt = _generator(backend=backend, selector=selector).build_prompt([_inspiration()])

    assert "Only the code between" in prompt
    assert "Keep marker lines exactly as written" in prompt
    assert "Return one text code block that includes both EVOLVE-BLOCK markers" in prompt
    assert "EXACT_PREFIX" in prompt
    assert "EXACT_SUFFIX" in prompt
    assert "Return exactly one JSON object" not in prompt
