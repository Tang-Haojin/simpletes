"""Engine-level regression coverage for persisted Codex repair artifacts."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from simpletes.engine.core import SimpleTESEngine
from simpletes.generator import GenerationTask, Generator
from simpletes.llm import LLMCallError


def test_generate_batch_records_repair_artifacts_with_generation_context():
    artifact_paths = (
        "/tmp/simpletes-test/request/attempt-01.json",
        "/tmp/simpletes-test/request/attempt-02.json",
    )

    class FailingGenerator:
        async def generate(self, task, instance_id, track_io):
            assert task.gen_id == 73
            assert instance_id == "test-instance"
            assert track_io is True
            raise LLMCallError(
                model="k3",
                api_base="https://example.invalid/v1",
                error_type="InvalidJSON",
                message="invalid JSON at line 2, column 7",
                details={"line": 2, "column": 7, "char_offset": 9},
                artifact_paths=artifact_paths,
            )

    class RecordingSelector:
        def __init__(self):
            self.failed_gen_ids: list[int] = []

        def on_generation_failed(self, gen_id):
            self.failed_gen_ids.append(gen_id)
            return None

    async def run_scenario():
        engine = object.__new__(SimpleTESEngine)
        engine.config = SimpleNamespace(
            save_llm_io=True,
            reflection_mode=False,
            llm_backend="codex_exec",
        )
        engine.generator = FailingGenerator()
        engine.selector = RecordingSelector()
        engine.instance_id = "test-instance"
        engine._counter_lock = asyncio.Lock()
        engine._gen_inflight = 0
        engine.generation_failures = 0
        engine._failure_records = []

        task = GenerationTask(
            prompt="repair this candidate",
            inspiration_ids=[],
            k=1,
            chain_idx=4,
            gen_id=73,
            shared_construction_id="shared-19",
        )
        await engine._generate_batch(task)
        return engine

    engine = asyncio.run(run_scenario())

    assert engine.generation_failures == 1
    assert engine._gen_inflight == 0
    assert engine.selector.failed_gen_ids == [73]
    assert len(engine._failure_records) == 1

    record = engine._failure_records[0]
    assert record["type"] == "generation"
    assert record["gen_id"] == 73
    assert record["chain_idx"] == 4
    assert record["shared_construction_id"] == "shared-19"
    assert record["artifact_paths"] == list(artifact_paths)
    assert record["error_type"] == "InvalidJSON"
    assert record["error_details"] == {
        "line": 2,
        "column": 7,
        "char_offset": 9,
    }
    assert json.loads(record["llm_output"]) == {
        "simpletes_codex_rejected_response_artifacts": list(artifact_paths)
    }


def test_generator_passes_gen_and_chain_context_to_llm_backend():
    observed: dict[str, object] = {}

    class RecordingBackend:
        async def generate_batch(
            self, prompt, n, instance_id="", track_io=False
        ):
            observed.update(
                prompt=prompt,
                n=n,
                instance_id=instance_id,
                track_io=track_io,
            )
            return []

    async def run_scenario():
        generator = object.__new__(Generator)
        generator._llm = RecordingBackend()
        task = GenerationTask(
            prompt="grounded request",
            inspiration_ids=[],
            k=1,
            chain_idx=3,
            gen_id=41,
        )
        assert await generator.generate(task, "instance-abcd", True) == []

    asyncio.run(run_scenario())

    assert observed == {
        "prompt": "grounded request",
        "n": 1,
        "instance_id": "instance-abcd-gen41-chain3",
        "track_io": True,
    }


def test_codex_error_without_artifact_is_still_checkpointed():
    class FailingGenerator:
        async def generate(self, _task, _instance_id, _track_io):
            raise LLMCallError(
                model="k3",
                api_base=None,
                error_type="CodexAuditError",
                message="artifact directory unavailable",
            )

    class RecordingSelector:
        def on_generation_failed(self, _gen_id):
            return None

    async def run_scenario():
        engine = object.__new__(SimpleTESEngine)
        engine.config = SimpleNamespace(
            save_llm_io=True,
            reflection_mode=False,
            llm_backend="codex_exec",
        )
        engine.generator = FailingGenerator()
        engine.selector = RecordingSelector()
        engine.instance_id = "test-instance"
        engine._counter_lock = asyncio.Lock()
        engine._gen_inflight = 0
        engine.generation_failures = 0
        engine._failure_records = []
        await engine._generate_batch(
            GenerationTask(
                prompt="prompt",
                inspiration_ids=[],
                k=1,
                chain_idx=0,
                gen_id=8,
            )
        )
        return engine

    engine = asyncio.run(run_scenario())

    assert len(engine._failure_records) == 1
    assert engine._failure_records[0]["artifact_paths"] == []
    assert engine._failure_records[0]["llm_output"] is None
