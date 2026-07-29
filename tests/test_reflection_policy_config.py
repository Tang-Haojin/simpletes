import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simpletes.engine.checkpoint import CheckpointManager
from simpletes.engine.core import SimpleTESEngine
from simpletes.cli import build_parser
from simpletes.config import EngineConfig
from simpletes.node import Node, NodeDatabase, Status
from simpletes.policies import PendingFinalize, create_selector


REFLECTION_POLICIES = [
    "balance",
    "puct",
    "rpucg",
    "llm_puct",
    "llm_rpucg",
    "llm_elite",
]


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_main_cli_accepts_extended_reasoning_efforts(effort):
    args = build_parser(mode="single").parse_args(
        ["--reasoning-effort", effort]
    )

    assert args.reasoning_effort == effort


def _build_policy(name: str):
    return create_selector(
        name,
        num_chains=1,
        max_generations=4,
        k=1,
        c=1.0,
        gamma=0.8,
        reflection_mode=True,
        llm_policy_model="reflection-model",
        llm_policy_api_base="https://reflection.example/v1",
        llm_policy_api_key="secret-key",
        llm_policy_pool_size=2,
    )


@pytest.mark.parametrize("policy_name", REFLECTION_POLICIES)
def test_create_selector_propagates_reflection_config(policy_name):
    policy = _build_policy(policy_name)

    assert policy.reflection_mode is True
    assert policy.llm_policy_model == "reflection-model"
    assert policy.llm_policy_api_base == "https://reflection.example/v1"

    state = policy.state_dict()
    assert state["reflection_mode"] is True
    assert state["llm_policy_model"] == "reflection-model"
    assert state["llm_policy_api_base"] == "https://reflection.example/v1"
    assert "llm_policy_api_key" not in state


@pytest.mark.parametrize("policy_name", REFLECTION_POLICIES)
def test_finalize_batch_sets_reflection_for_batch_best(policy_name, monkeypatch):
    policy = _build_policy(policy_name)
    db = NodeDatabase()

    root = Node(
        id="root",
        code="def solve():\n    return 0\n",
        metrics={"combined_score": 0.0},
        score=0.0,
        status=Status.DONE,
    )
    child = Node(
        id="child",
        code="def solve():\n    return 1\n",
        parent_ids=["root"],
        metrics={"combined_score": 1.0},
        score=1.0,
        status=Status.DONE,
        llm_input="Improve the program",
    )
    db.add(root)
    db.add(child)

    async def fake_llm_generate(messages, temperature=0.7, max_tokens=None):
        del messages, temperature, max_tokens
        return SimpleNamespace(
            content="Approach: try a better heuristic\nInsight: keep the stronger path"
        )

    monkeypatch.setattr(policy, "_llm_generate", fake_llm_generate)

    policy.register_batch(gen_id=7, chain_idx=0, inspiration_ids=["root"], k=1)
    pending = PendingFinalize(
        gen_id=7,
        chain_idx=0,
        children=[("child", None)],
        inspirations=["root"],
    )

    completion = asyncio.run(policy.finalize_batch(pending, db))

    assert completion.best_node_id == "child"
    assert child.reflection == "Approach: try a better heuristic\nInsight: keep the stronger path"
    assert "child" in policy.chains[0]


def test_reflect_on_winner_accepts_raw_string_llm_response(monkeypatch):
    policy = _build_policy("balance")
    node = Node(
        id="child",
        code="def solve():\n    return 1\n",
        metrics={"combined_score": 1.0},
        score=1.0,
        status=Status.DONE,
        llm_input="Improve the program",
    )

    async def fake_llm_generate(messages, temperature=0.7, max_tokens=None):
        del messages, temperature, max_tokens
        return "Approach: direct string\nInsight: still parsed"

    monkeypatch.setattr(policy, "_llm_generate", fake_llm_generate)

    reflection = asyncio.run(policy.reflect_on_winner(node))

    assert reflection == "Approach: direct string\nInsight: still parsed"


def test_reflect_on_winner_returns_empty_for_missing_content(monkeypatch):
    policy = _build_policy("balance")
    node = Node(
        id="child",
        code="def solve():\n    return 1\n",
        metrics={"combined_score": 1.0},
        score=1.0,
        status=Status.DONE,
        llm_input="Improve the program",
    )

    async def fake_llm_generate(messages, temperature=0.7, max_tokens=None):
        del messages, temperature, max_tokens
        return SimpleNamespace(content=None)

    monkeypatch.setattr(policy, "_llm_generate", fake_llm_generate)

    assert asyncio.run(policy.reflect_on_winner(node)) == ""


def test_checkpoint_config_serializes_effective_llm_policy_values(tmp_path):
    config = EngineConfig(
        init_program="init.py",
        evaluator_path="eval.py",
        instruction_path="prompt.txt",
        model="generator-model",
        api_base="https://generator.example/v1",
        llm_backend="vllm_token_forcing",
        context_window=32768,
        reasoning_budget=26000,
        response_budget=6768,
        reflection_mode=True,
    )
    manager = CheckpointManager(config, "instance-id", str(tmp_path))

    serialized = manager._config_to_dict()

    assert serialized["reflection_mode"] is True
    assert serialized["llm_policy_model"] == "generator-model"
    assert serialized["llm_policy_api_base"] == "https://generator.example/v1"
    assert serialized["llm_backend"] == "vllm_token_forcing"
    assert serialized["context_window"] == 32768
    assert serialized["reasoning_budget"] == 26000
    assert serialized["response_budget"] == 6768


def test_checkpoint_config_persists_k3_generation_provenance_without_auth(
    tmp_path, monkeypatch
):
    config = EngineConfig(
        init_program="init.py",
        evaluator_path="eval.py",
        instruction_path="prompt.txt",
        instruction_suffix_path="prompt.k3.txt",
        model="k3",
        reasoning_effort="ultra",
        llm_backend="codex_exec",
        api_key="api-key-must-not-persist",
        codex_config_path="/safe/config.kimi.toml",
        codex_auth_path="/secret/auth-path-must-not-persist.json",
        codex_repo_root="/work/grhsim-repo",
        codex_output_schema="/work/candidate.schema.json",
        codex_local_validation_schema="/work/candidate.local.schema.json",
        codex_output_mode="local-json",
        codex_tool_choice_mode="required-first",
        codex_max_agent_threads=3,
        codex_model_catalog_path="/work/k3_model_catalog.json",
        llm_policy_api_key="policy-key-must-not-persist",
    )
    manager = CheckpointManager(config, "instance-id", str(tmp_path))
    monkeypatch.setattr(
        "simpletes.engine.checkpoint.save_score_statistics",
        lambda *_args, **_kwargs: None,
    )

    serialized = manager._config_to_dict()
    manager.write_sync(
        best_code=None,
        metadata={"completed_evaluations": 0, "best_score": 0.0},
        config=serialized,
        policy={"name": "rpucg", "state": {}},
        nodes=[],
        failure_records=[],
    )

    config_paths = list(tmp_path.glob("db_state_*/config.json"))
    assert len(config_paths) == 1
    on_disk = json.loads(config_paths[0].read_text(encoding="utf-8"))
    assert on_disk["model"] == "k3"
    assert on_disk["instruction_suffix_path"] == "prompt.k3.txt"
    assert on_disk["reasoning_effort"] == "ultra"
    assert on_disk["llm_backend"] == "codex_exec"
    assert on_disk["codex_config_path"] == "/safe/config.kimi.toml"
    assert on_disk["codex_repo_root"] == "/work/grhsim-repo"
    assert on_disk["codex_output_schema"] == "/work/candidate.schema.json"
    assert on_disk["codex_local_validation_schema"] == (
        "/work/candidate.local.schema.json"
    )
    assert on_disk["codex_output_mode"] == "local-json"
    assert on_disk["codex_tool_choice_mode"] == "required-first"
    assert on_disk["codex_max_agent_threads"] == 3
    assert on_disk["codex_model_catalog_path"] == "/work/k3_model_catalog.json"

    rendered = json.dumps(on_disk, sort_keys=True)
    assert "codex_auth_path" not in on_disk
    assert "api_key" not in on_disk
    assert "llm_policy_api_key" not in on_disk
    assert "auth-path-must-not-persist" not in rendered
    assert "api-key-must-not-persist" not in rendered
    assert "policy-key-must-not-persist" not in rendered


def test_checkpoint_load_accepts_legacy_config_without_codex_provenance(tmp_path):
    state = tmp_path / "db_state_legacy"
    state.mkdir()
    (state / "metadata.json").write_text(
        json.dumps({"instance_id": "legacy", "completed_evaluations": 0}),
        encoding="utf-8",
    )
    (state / "config.json").write_text(
        json.dumps({"model": "k3", "llm_backend": "codex_exec"}),
        encoding="utf-8",
    )
    (state / "policy.json").write_text(
        json.dumps({"name": "balance", "state": {}}),
        encoding="utf-8",
    )
    (state / "nodes.json").write_text("[]\n", encoding="utf-8")

    config = EngineConfig(
        init_program="init.py",
        evaluator_path="eval.py",
        instruction_path="prompt.txt",
    )
    manager = CheckpointManager(config, "current", str(tmp_path / "output"))
    restored = manager.load(str(state), NodeDatabase(), _build_policy("balance"))

    assert restored["instance_id"] == "legacy"
    assert restored["completed_evaluations"] == 0
    assert restored["checkpoint_config"] == {
        "model": "k3",
        "llm_backend": "codex_exec",
    }
    assert restored["budget_extensions"] == []


def _exhausted_rpucg_policy():
    policy = create_selector(
        "rpucg",
        num_chains=4,
        max_generations=64,
        k=1,
        c=0.5,
        gamma=0.8,
    )
    state = policy.state_dict()
    state["chain_prompt_count"] = {"0": 16, "1": 16, "2": 7, "3": 16}
    policy.load_state_dict(state)
    return policy


def test_trajectory_policy_monotonically_extends_restored_chain_budgets():
    policy = _exhausted_rpucg_policy()

    extension = policy.extend_generation_budget(192)

    assert extension["previous_max_generations"] == 64
    assert extension["new_max_generations"] == 192
    assert policy.max_generations == 192
    assert policy.chain_gen_budget == {0: 48, 1: 48, 2: 48, 3: 48}
    assert policy.prompt_budget == {0: 48, 1: 48, 2: 48, 3: 48}
    assert policy.chain_prompt_count == {0: 16, 1: 16, 2: 7, 3: 16}
    assert policy._ready_chains == [0, 1, 2, 3]

    with pytest.raises(ValueError, match="must be greater"):
        policy.extend_generation_budget(192)


def test_engine_resume_budget_extension_is_explicit_and_audited(tmp_path):
    checkpoint_config = {
        "max_generations": 64,
        "max_valid_evaluations": 32,
    }
    policy = _exhausted_rpucg_policy()
    config = EngineConfig(
        init_program="init.py",
        evaluator_path="eval.py",
        instruction_path="prompt.txt",
        selector="rpucg",
        num_chains=4,
        k_candidates=1,
        max_generations=192,
        max_valid_evaluations=32,
        extend_resume_budget=True,
    )
    engine = SimpleNamespace(
        config=config,
        selector=policy,
        generation_attempts=64,
        valid_evaluations=15,
        _budget_extensions=[],
    )

    record = SimpleTESEngine._apply_resume_budget_contract(
        engine,
        str(tmp_path / "db_state_143702"),
        {"checkpoint_config": checkpoint_config},
    )

    assert record is not None
    assert record["previous_max_generations"] == 64
    assert record["new_max_generations"] == 192
    assert record["generation_attempts_at_extension"] == 64
    assert record["valid_evaluations_at_extension"] == 15
    assert record["policy"]["new_prompt_budget"] == {
        0: 48,
        1: 48,
        2: 48,
        3: 48,
    }
    assert engine._budget_extensions == [record]


def test_engine_resume_budget_change_fails_without_explicit_extension(tmp_path):
    policy = _exhausted_rpucg_policy()
    config = EngineConfig(
        init_program="init.py",
        evaluator_path="eval.py",
        instruction_path="prompt.txt",
        selector="rpucg",
        num_chains=4,
        k_candidates=1,
        max_generations=192,
        max_valid_evaluations=32,
    )
    engine = SimpleNamespace(
        config=config,
        selector=policy,
        generation_attempts=64,
        valid_evaluations=15,
        _budget_extensions=[],
    )

    with pytest.raises(ValueError, match="--extend-resume-budget"):
        SimpleTESEngine._apply_resume_budget_contract(
            engine,
            str(tmp_path / "db_state_143702"),
            {
                "checkpoint_config": {
                    "max_generations": 64,
                    "max_valid_evaluations": 32,
                }
            },
        )

    assert policy.max_generations == 64
    assert policy.prompt_budget == {0: 16, 1: 16, 2: 16, 3: 16}
