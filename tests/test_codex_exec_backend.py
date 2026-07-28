from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

from simpletes.cli import build_parser
from simpletes.config import EngineConfig, build_config_from_args
from simpletes.llm import create_llm_client
from simpletes.llm.codex_exec import CodexExecClient
from simpletes.llm.types import LLMCallError


SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema_version", "patch"],
    "properties": {
        "schema_version": {"const": 1},
        "patch": {"type": "string"},
    },
}
DEFAULT_EVENTS = [
    {"type": "thread.started", "thread_id": "fake-thread"},
    {
        "type": "item.started",
        "item": {"id": "repo-read-1", "type": "command_execution"},
    },
    {
        "type": "item.completed",
        "item": {"id": "repo-read-1", "type": "command_execution"},
    },
]
LOCAL_VALIDATION_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["patch"],
    "properties": {
        "patch": {
            "type": "string",
            "minLength": 1,
            "not": {"enum": ["placeholder"]},
            "pattern": "\\n$",
        }
    },
}


def _make_inputs(tmp_path: Path) -> dict[str, Path]:
    config = tmp_path / "source.toml"
    config.write_text(
        'model_provider = "OpenAI"\n'
        'model = "gpt-5.6-sol"\n'
        'model_reasoning_effort = "ultra"\n'
        '[model_providers.OpenAI]\n'
        'name = "OpenAI"\n',
        encoding="utf-8",
    )
    auth = tmp_path / "source-auth.json"
    auth.write_text('{"OPENAI_API_KEY": "unit-test-secret"}\n', encoding="utf-8")
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(SCHEMA), encoding="utf-8")
    local_schema = tmp_path / "local-validation-schema.json"
    local_schema.write_text(json.dumps(LOCAL_VALIDATION_SCHEMA), encoding="utf-8")
    model_catalog = tmp_path / "model-catalog.json"
    model_catalog.write_text(
        json.dumps({"models": [{"slug": "gpt-5.6-sol"}]}),
        encoding="utf-8",
    )
    repo = tmp_path / "repo"
    repo.mkdir()

    executable = tmp_path / "fake-codex"
    settings = tmp_path / "fake-settings.json"
    settings.write_text(
        json.dumps(
            {
                "response": {"schema_version": 1, "patch": ""},
                "raw": None,
                "stderr": "",
                "status": 0,
                "events": DEFAULT_EVENTS,
            }
        ),
        encoding="utf-8",
    )
    executable.write_text(
        """#!/usr/bin/env python3
import json
import fcntl
import os
import pathlib
import stat
import sys
import time

args = sys.argv[1:]
base = pathlib.Path(__file__).resolve().parent
settings = json.loads((base / "fake-settings.json").read_text())
counter_path = base / "invocation-counter"
with counter_path.open("a+") as counter:
    fcntl.flock(counter, fcntl.LOCK_EX)
    counter.seek(0)
    raw_counter = counter.read().strip()
    invocation_index = int(raw_counter) if raw_counter else 0
    counter.seek(0)
    counter.truncate()
    counter.write(str(invocation_index + 1))
    counter.flush()
attempts = settings.get("attempts")
if attempts:
    settings = attempts[min(invocation_index, len(attempts) - 1)]
home = pathlib.Path(os.environ["CODEX_HOME"])
record = {
    "args": args,
    "home": str(home),
    "home_mode": stat.S_IMODE(home.stat().st_mode),
    "config_mode": stat.S_IMODE((home / "config.toml").stat().st_mode),
    "model_catalog_exists": (home / "model_catalog.json").exists(),
    "model_catalog_mode": (
        stat.S_IMODE((home / "model_catalog.json").stat().st_mode)
        if (home / "model_catalog.json").exists()
        else None
    ),
    "model_catalog_text": (
        (home / "model_catalog.json").read_text()
        if (home / "model_catalog.json").exists()
        else None
    ),
    "auth_exists": (home / "auth.json").exists(),
    "inherited_api_key": os.environ.get("OPENAI_API_KEY"),
    "inherited_webhook": os.environ.get("WX_WEBHOOK_URL"),
    "inherited_github_token": os.environ.get("GITHUB_TOKEN"),
    "inherited_arbitrary": os.environ.get("ARBITRARY_PARENT_VALUE"),
    "prompt": sys.stdin.read(),
}
(base / "invocation.json").write_text(json.dumps(record))
(base / f"invocation-{invocation_index}.json").write_text(json.dumps(record))
for event in settings.get("events", []):
    print(json.dumps(event), flush=True)
time.sleep(float(settings.get("sleep", 0)))
sys.stderr.write(settings["stderr"])
status = int(settings["status"])
if status == 0:
    output = pathlib.Path(args[args.index("-o") + 1])
    output.write_text(settings["raw"] if settings["raw"] is not None else json.dumps(settings["response"]))
sys.exit(status)
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return {
        "config": config,
        "auth": auth,
        "schema": schema,
        "local_schema": local_schema,
        "model_catalog": model_catalog,
        "repo": repo,
        "executable": executable,
        "settings": settings,
        "log": tmp_path / "invocation.json",
        "counter": tmp_path / "invocation-counter",
    }


def _set_fake(
    paths: dict[str, Path],
    *,
    response: object = None,
    raw: str | None = None,
    status: int = 0,
    stderr: str = "",
    events: list[dict[str, object]] | None = None,
    sleep: float = 0,
) -> None:
    paths["settings"].write_text(
        json.dumps(
            {
                "response": response,
                "raw": raw,
                "stderr": stderr,
                "status": status,
                "events": DEFAULT_EVENTS if events is None else events,
                "sleep": sleep,
            }
        ),
        encoding="utf-8",
    )


def _set_fake_sequence(
    paths: dict[str, Path], attempts: list[dict[str, object]]
) -> None:
    normalized = []
    for attempt in attempts:
        normalized.append(
            {
                "response": attempt.get("response"),
                "raw": attempt.get("raw"),
                "stderr": attempt.get("stderr", ""),
                "status": attempt.get("status", 0),
                "events": attempt.get("events", DEFAULT_EVENTS),
            }
        )
    paths["settings"].write_text(
        json.dumps({"attempts": normalized}), encoding="utf-8"
    )


def _client(paths: dict[str, Path], **overrides) -> CodexExecClient:
    kwargs = {
        "model": "gpt-5.6-sol",
        "reasoning_effort": "ultra",
        "config_path": str(paths["config"]),
        "auth_path": str(paths["auth"]),
        "repo_root": str(paths["repo"]),
        "output_schema": str(paths["schema"]),
        "timeout": 10,
        "codex_binary": str(paths["executable"]),
    }
    kwargs.update(overrides)
    return CodexExecClient(**kwargs)


def test_generate_uses_isolated_home_and_structured_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(paths, response={"schema_version": 1, "patch": "diff --git"})
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-inherited")
    monkeypatch.setenv("WX_WEBHOOK_URL", "must-not-be-inherited")
    monkeypatch.setenv("GITHUB_TOKEN", "must-not-be-inherited")
    monkeypatch.setenv("ARBITRARY_PARENT_VALUE", "must-not-be-inherited")

    result = asyncio.run(_client(paths).generate("inspect repo", track_io=True))
    record = json.loads(paths["log"].read_text(encoding="utf-8"))

    assert result.text.startswith("```python\n# EVOLVE-BLOCK-START\n{")
    assert result.text.endswith("\n# EVOLVE-BLOCK-END\n```")
    assert json.loads(result.raw_output or "") == {
        "schema_version": 1,
        "patch": "diff --git",
    }
    assert result.prompt == "inspect repo"
    assert record["prompt"] == "inspect repo"
    assert record["home_mode"] == 0o700
    assert record["config_mode"] == 0o600
    assert not record["auth_exists"]
    assert record["inherited_api_key"] == "unit-test-secret"
    assert record["inherited_webhook"] is None
    assert record["inherited_github_token"] is None
    assert record["inherited_arbitrary"] is None
    assert not Path(record["home"]).exists()

    args = record["args"]
    assert args[:3] == ["-a", "never", "exec"]
    assert "--ephemeral" in args
    assert args[args.index("--disable") + 1] == "plugins"
    disabled_features = {
        args[index + 1]
        for index, value in enumerate(args[:-1])
        if value == "--disable"
    }
    assert disabled_features == {"plugins"}
    assert "--ignore-rules" in args
    assert "--json" in args
    assert args[args.index("--output-schema") + 1] == str(paths["schema"])
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert args[args.index("-C") + 1] == str(paths["repo"])
    assert args[args.index("-m") + 1] == "gpt-5.6-sol"
    assert 'model_reasoning_effort="ultra"' in args
    assert "model_providers.OpenAI.requires_openai_auth=false" in args
    assert 'model_providers.OpenAI.env_key="OPENAI_API_KEY"' in args
    assert 'network_access="disabled"' in args
    assert 'shell_environment_policy.inherit="none"' in args
    assert any(value.startswith("shell_environment_policy.set.PATH=") for value in args)
    assert any(value.startswith("shell_environment_policy.set.HOME=") for value in args)
    assert "mcp_servers={}" in args
    assert "notify=[]" in args
    assert str(paths["auth"]) not in args
    assert "unit-test-secret" not in json.dumps(args)


@pytest.mark.parametrize(
    "response,error_type",
    [
        ("not-json", "InvalidJSON"),
        ({"schema_version": 2, "patch": "x"}, "SchemaValidationError"),
        ({"schema_version": 1}, "SchemaValidationError"),
    ],
)
def test_generate_rejects_invalid_structured_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: object,
    error_type: str,
) -> None:
    paths = _make_inputs(tmp_path)
    if response == "not-json":
        _set_fake(paths, raw="not-json")
    else:
        _set_fake(paths, response=response)

    with pytest.raises(LLMCallError) as raised:
        asyncio.run(_client(paths).generate("prompt"))
    assert raised.value.error_type == error_type
    assert "exhausted 3 attempt(s)" in raised.value.message
    assert paths["counter"].read_text(encoding="utf-8") == "3"


def test_generate_repairs_schema_failure_and_saves_sanitized_trace(
    tmp_path: Path,
) -> None:
    paths = _make_inputs(tmp_path)
    raw_event_payload = "private-repository-output-must-not-be-saved"
    _set_fake_sequence(
        paths,
        [
            {
                "response": {"schema_version": 1},
                "events": [
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "repo-read-1",
                            "type": "command_execution",
                            "aggregated_output": raw_event_payload,
                        },
                    }
                ],
            },
            {"response": {"schema_version": 1, "patch": "real diff\n"}},
        ],
    )

    result = asyncio.run(_client(paths).generate("inspect repo", track_io=True))
    tracked = json.loads(result.raw_output or "")
    repair_prompt = json.loads(
        (tmp_path / "invocation-1.json").read_text(encoding="utf-8")
    )["prompt"]

    assert tracked["response"]["patch"] == "real diff\n"
    audit = tracked["simpletes_codex_audit"]
    assert audit["attempt_count"] == 2
    assert len(audit["failure_summaries"]) == 1
    assert "required property" in audit["failure_summaries"][0]
    assert "Exact validation error:" in repair_prompt
    assert "'patch' is a required property" in repair_prompt
    assert repair_prompt.startswith("inspect repo\n\n")
    assert raw_event_payload not in (result.raw_output or "")


def test_local_validation_overlay_repairs_provider_schema_valid_response(
    tmp_path: Path,
) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake_sequence(
        paths,
        [
            {"response": {"schema_version": 1, "patch": ""}},
            {"response": {"schema_version": 1, "patch": "real diff\n"}},
        ],
    )

    result = asyncio.run(
        _client(
            paths,
            local_validation_schema=str(paths["local_schema"]),
        ).generate("inspect repo", track_io=True)
    )
    tracked = json.loads(result.raw_output or "")
    invocation = json.loads(paths["log"].read_text(encoding="utf-8"))
    repair_prompt = json.loads(
        (tmp_path / "invocation-1.json").read_text(encoding="utf-8")
    )["prompt"]

    assert tracked["response"]["patch"] == "real diff\n"
    assert "SemanticValidationError" in tracked["simpletes_codex_audit"][
        "failure_summaries"
    ][0]
    assert "local semantic validation at patch" in repair_prompt
    args = invocation["args"]
    assert args[args.index("--output-schema") + 1] == str(paths["schema"])
    assert str(paths["local_schema"]) not in args


def test_local_json_mode_omits_provider_schema_but_repairs_both_local_layers(
    tmp_path: Path,
) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake_sequence(
        paths,
        [
            {"response": {"schema_version": 2, "patch": "first diff\n"}},
            {"response": {"schema_version": 1, "patch": ""}},
            {"response": {"schema_version": 1, "patch": "final diff\n"}},
        ],
    )
    client = _client(
        paths,
        output_mode="local-json",
        local_validation_schema=str(paths["local_schema"]),
    )

    result = asyncio.run(client.generate("inspect repo", track_io=True))
    tracked = json.loads(result.raw_output or "")
    invocation = json.loads(paths["log"].read_text(encoding="utf-8"))
    failures = tracked["simpletes_codex_audit"]["failure_summaries"]

    assert tracked["response"]["patch"] == "final diff\n"
    assert len(failures) == 2
    assert "SchemaValidationError" in failures[0]
    assert "SemanticValidationError" in failures[1]
    assert "--output-schema" not in invocation["args"]
    assert str(paths["schema"]) not in invocation["args"]


@pytest.mark.parametrize("operation", ["generate", "preflight", "capability"])
def test_local_json_mode_is_used_by_every_codex_operation(
    tmp_path: Path,
    operation: str,
) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(
        paths,
        response={"schema_version": 1, "patch": "valid diff\n"},
    )
    client = _client(paths, output_mode="local-json")

    if operation == "generate":
        asyncio.run(client.generate("prompt"))
    elif operation == "preflight":
        asyncio.run(client.preflight("prompt"))
    else:
        asyncio.run(client.capability_probe("prompt"))
    invocation = json.loads(paths["log"].read_text(encoding="utf-8"))

    assert "--output-schema" not in invocation["args"]


def test_local_validation_overlay_applies_to_preflight(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake_sequence(
        paths,
        [
            {"response": {"schema_version": 1, "patch": "placeholder"}},
            {"response": {"schema_version": 1, "patch": "grounded\n"}},
        ],
    )
    client = _client(
        paths,
        local_validation_schema=str(paths["local_schema"]),
    )

    result = asyncio.run(client.preflight("return a concrete response"))

    assert result.value["patch"] == "grounded\n"
    assert result.trace.attempt_count == 2
    assert "SemanticValidationError" in result.trace.failure_summaries[0]


def test_provider_schema_validation_precedes_local_overlay(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(
        paths,
        response={"schema_version": 2, "patch": ""},
    )

    with pytest.raises(LLMCallError) as raised:
        asyncio.run(
            _client(
                paths,
                local_validation_schema=str(paths["local_schema"]),
                max_repair_attempts=0,
            ).generate("prompt")
        )

    assert raised.value.error_type == "SchemaValidationError"
    assert "local semantic validation" not in raised.value.message


@pytest.mark.parametrize(
    "schema_text",
    ["not-json", '{"type": "not-a-json-schema-type"}'],
)
def test_local_validation_overlay_must_be_valid_draft_2020_12_schema(
    tmp_path: Path,
    schema_text: str,
) -> None:
    paths = _make_inputs(tmp_path)
    paths["local_schema"].write_text(schema_text, encoding="utf-8")

    with pytest.raises(ValueError, match="not a valid JSON Schema"):
        _client(
            paths,
            local_validation_schema=str(paths["local_schema"]),
        )


def test_local_validation_overlay_symbolic_link_is_rejected(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    overlay_link = tmp_path / "overlay-link.json"
    overlay_link.symlink_to(paths["local_schema"])

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        _client(paths, local_validation_schema=str(overlay_link))


def test_local_validation_overlay_must_be_regular_file(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    overlay_directory = tmp_path / "overlay-directory"
    overlay_directory.mkdir()

    with pytest.raises(ValueError, match="must be a regular file"):
        _client(paths, local_validation_schema=str(overlay_directory))


def test_generate_repairs_semantically_rejected_response(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake_sequence(
        paths,
        [
            {"response": {"schema_version": 1, "patch": "placeholder"}},
            {"response": {"schema_version": 1, "patch": "real diff\n"}},
        ],
    )

    def reject_placeholder(value, trace):
        del trace
        return "patch must not be placeholder" if value["patch"] == "placeholder" else None

    result = asyncio.run(
        _client(paths, response_validator=reject_placeholder).generate(
            "inspect repo", track_io=True
        )
    )
    tracked = json.loads(result.raw_output or "")
    assert tracked["response"]["patch"] == "real diff\n"
    assert "SemanticValidationError" in tracked["simpletes_codex_audit"][
        "failure_summaries"
    ][0]


def test_repair_prompt_and_trace_redact_secret_shaped_text(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    secret_shaped = "sk-provider-accidental-secret"
    _set_fake_sequence(
        paths,
        [
            {
                "response": {
                    "schema_version": 2,
                    "patch": secret_shaped,
                }
            },
            {"response": {"schema_version": 1, "patch": "real diff\n"}},
        ],
    )

    result = asyncio.run(_client(paths).generate("prompt", track_io=True))
    repair_prompt = json.loads(
        (tmp_path / "invocation-1.json").read_text(encoding="utf-8")
    )["prompt"]

    assert secret_shaped not in repair_prompt
    assert "[REDACTED]" in repair_prompt
    assert secret_shaped not in (result.raw_output or "")


def test_capability_probe_returns_repo_tool_trace_and_canonical_value(
    tmp_path: Path,
) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(paths, response={"schema_version": 1, "patch": "grounded\n"})

    result = asyncio.run(_client(paths).capability_probe("read a pinned source"))

    assert result.value == {"schema_version": 1, "patch": "grounded\n"}
    assert json.loads(result.canonical) == result.value
    assert result.trace.attempt_count == 1
    assert result.trace.event_count == 3
    assert result.trace.repo_tool_call_count == 1
    assert result.trace.repo_tool_types == ("command_execution",)
    assert result.trace.failure_summaries == ()


def test_collaboration_trace_is_deduplicated_and_payload_free(
    tmp_path: Path,
) -> None:
    paths = _make_inputs(tmp_path)
    private_prompt = "private-child-task-must-not-be-saved"
    private_message = "private-child-message-must-not-be-saved"
    _set_fake(
        paths,
        response={"schema_version": 1, "patch": "collaborative\n"},
        events=[
            {
                "type": "item.started",
                "item": {
                    "id": "collab-1",
                    "type": "collab_tool_call",
                    "tool": "spawn_agent",
                    "prompt": private_prompt,
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "collab-1",
                    "type": "collab_tool_call",
                    "tool": "spawn_agent",
                    "agents_states": {
                        "child": {"status": "completed", "message": private_message}
                    },
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "collab-2",
                    "type": "collab_tool_call",
                    "tool": "send_input",
                    "prompt": private_prompt,
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "id": "collab-3",
                    "type": "collab_tool_call",
                    "tool": "wait",
                },
            },
        ],
    )

    result = asyncio.run(
        _client(paths).generate("collaborate", track_io=True)
    )
    tracked = json.loads(result.raw_output or "")
    audit = tracked["simpletes_codex_audit"]

    assert audit["collaboration_tool_call_count"] == 3
    assert audit["collaboration_tool_call_counts"] == {
        "send_input": 1,
        "spawn_agent": 1,
        "wait": 1,
    }
    assert private_prompt not in (result.raw_output or "")
    assert private_message not in (result.raw_output or "")


def test_capability_probe_rejects_response_without_repo_tool_call(
    tmp_path: Path,
) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(
        paths,
        response={"schema_version": 1, "patch": "ungrounded\n"},
        events=[{"type": "thread.started", "thread_id": "fake-thread"}],
    )

    with pytest.raises(LLMCallError) as raised:
        asyncio.run(
            _client(paths, max_repair_attempts=0).capability_probe("inspect repo")
        )

    assert raised.value.error_type == "SemanticValidationError"
    assert "no repository tool call" in raised.value.message
    assert "repo_tool_call_count\": 0" in raised.value.message


def test_capability_probe_requires_repo_tool_in_final_successful_attempt(
    tmp_path: Path,
) -> None:
    paths = _make_inputs(tmp_path)
    no_tools = [{"type": "thread.started", "thread_id": "fake-thread"}]
    _set_fake_sequence(
        paths,
        [
            {
                "response": {"schema_version": 1},
                "events": DEFAULT_EVENTS,
            },
            {
                "response": {"schema_version": 1, "patch": "ungrounded\n"},
                "events": no_tools,
            },
            {
                "response": {"schema_version": 1, "patch": "grounded\n"},
                "events": DEFAULT_EVENTS,
            },
        ],
    )

    result = asyncio.run(_client(paths).capability_probe("inspect repo"))

    assert result.value["patch"] == "grounded\n"
    assert result.trace.attempt_count == 3
    assert result.trace.repo_tool_call_count == 2
    assert "no repository tool call" in result.trace.failure_summaries[1]


def test_capability_probe_semantic_callback_exhaustion_is_bounded_and_redacted(
    tmp_path: Path,
) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(
        paths,
        response={"schema_version": 1, "patch": "grounded\n"},
    )

    def always_reject(value, trace):
        del value, trace
        return "candidate leaked sk-semantic-callback-secret"

    with pytest.raises(LLMCallError) as raised:
        asyncio.run(
            _client(paths, max_repair_attempts=1).capability_probe(
                "inspect repo", validate=always_reject
            )
        )

    assert raised.value.error_type == "SemanticValidationError"
    assert "exhausted 2 attempt(s)" in raised.value.message
    assert "sk-semantic-callback-secret" not in raised.value.message
    assert raised.value.message.count("SemanticValidationError") == 2
    assert paths["counter"].read_text(encoding="utf-8") == "2"


def test_generic_preflight_is_transport_only_without_repo_tool_trace(
    tmp_path: Path,
) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(
        paths,
        response={"schema_version": 1, "patch": "schema-only\n"},
        events=[{"type": "thread.started", "thread_id": "fake-thread"}],
    )

    result = asyncio.run(_client(paths, max_repair_attempts=0).preflight())
    assert result.value["patch"] == "schema-only\n"
    assert result.trace.repo_tool_call_count == 0


def test_isolated_home_cleanup_retries_short_lived_directory_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(paths, response={"schema_version": 1, "patch": "diff\n"})
    real_rmtree = shutil.rmtree
    cleanup_attempts = 0

    def flaky_rmtree(path, *args, **kwargs):
        nonlocal cleanup_attempts
        if Path(path).name.startswith("simpletes-codex-"):
            cleanup_attempts += 1
            if cleanup_attempts <= 2:
                raise OSError(39, "Directory not empty")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("simpletes.llm.codex_exec.shutil.rmtree", flaky_rmtree)

    asyncio.run(_client(paths).generate("prompt"))
    record = json.loads(paths["log"].read_text(encoding="utf-8"))
    assert cleanup_attempts == 3
    assert not Path(record["home"]).exists()


def test_generate_does_not_echo_failed_subprocess_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(
        paths,
        response={},
        status=7,
        stderr="provider accidentally printed unit-test-secret",
        events=[
            {
                "type": "item.completed",
                "item": {
                    "id": "repo-read-secret",
                    "type": "command_execution",
                    "command": "printf unit-test-secret",
                    "aggregated_output": "unit-test-secret",
                },
            }
        ],
    )

    with pytest.raises(LLMCallError) as raised:
        asyncio.run(_client(paths).generate("prompt"))
    assert raised.value.error_type == "CodexExecError"
    assert "unit-test-secret" not in str(raised.value)


def test_event_jsonl_is_read_from_a_bounded_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(
        paths,
        response={"schema_version": 1, "patch": "bounded\n"},
        events=[
            *DEFAULT_EVENTS,
            {"type": "oversized.test.event", "payload": "x" * 4096},
        ],
    )
    import simpletes.llm.codex_exec as codex_exec

    monkeypatch.setattr(codex_exec, "_MAX_CODEX_STDOUT_BYTES", 512)

    result = asyncio.run(_client(paths).capability_probe("inspect repo"))

    assert result.trace.repo_tool_call_count == 1
    assert any(
        "bounded audit limit" in diagnostic
        for diagnostic in result.trace.diagnostic_summaries
    )


def test_timeout_reports_bounded_repo_tool_trace(
    tmp_path: Path,
) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(
        paths,
        response={"schema_version": 1, "patch": "never-written\n"},
        events=[
            {
                "type": "item.completed",
                "item": {
                    "id": "repo-read-before-timeout",
                    "type": "command_execution",
                    "aggregated_output": "private-timeout-payload",
                },
            }
        ],
        sleep=2,
    )

    with pytest.raises(LLMCallError) as raised:
        asyncio.run(_client(paths, timeout=0.05).generate("inspect repo"))

    assert raised.value.error_type == "TimeoutError"
    assert '"repo_tool_call_count": 1' in raised.value.message
    assert '"repo_tool_types": ["command_execution"]' in raised.value.message
    assert "private-timeout-payload" not in raised.value.message


def test_failed_subprocess_uses_bounded_stderr_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(
        paths,
        response={},
        status=7,
        stderr="DROPPED-PREFIX-" + ("x" * 512) + "-TAIL-ERROR",
    )
    import simpletes.llm.codex_exec as codex_exec

    monkeypatch.setattr(codex_exec, "_MAX_CODEX_STDERR_BYTES", 64)

    with pytest.raises(LLMCallError) as raised:
        asyncio.run(_client(paths).generate("prompt"))

    assert raised.value.error_type == "CodexExecError"
    assert "TAIL-ERROR" in raised.value.message
    assert "DROPPED-PREFIX" not in raised.value.message
    assert "stderr tail (truncated)" in raised.value.message


def test_final_response_is_rejected_before_unbounded_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(
        paths,
        response={"schema_version": 1, "patch": "x" * 4096},
    )
    import simpletes.llm.codex_exec as codex_exec

    monkeypatch.setattr(codex_exec, "_MAX_FINAL_OUTPUT_BYTES", 128)

    with pytest.raises(LLMCallError) as raised:
        asyncio.run(_client(paths).generate("prompt"))

    assert raised.value.error_type == "OutputTooLarge"
    assert "bounded output limit" in raised.value.message


def test_process_group_kill_falls_back_to_direct_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import simpletes.llm.codex_exec as codex_exec

    class FakeProcess:
        pid = 4242
        returncode = None
        killed = False
        waited = False

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            self.waited = True
            self.returncode = -9
            return self.returncode

    process = FakeProcess()

    def fail_killpg(_pid: int, _signal: int) -> None:
        raise OSError("process group unavailable")

    monkeypatch.setattr(codex_exec.os, "killpg", fail_killpg)

    asyncio.run(CodexExecClient._kill_process(process))

    assert process.killed is True
    assert process.waited is True


def test_config_model_and_effort_must_match(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    with pytest.raises(ValueError, match="model mismatch"):
        _client(paths, model="different-model")
    with pytest.raises(ValueError, match="reasoning-effort mismatch"):
        _client(paths, reasoning_effort="high")


def test_codex_output_mode_must_be_known(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)

    with pytest.raises(ValueError, match="output_mode must be one of"):
        _client(paths, output_mode="unknown-mode")


def test_codex_tool_choice_mode_must_be_known(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)

    with pytest.raises(ValueError, match="tool_choice_mode must be one of"):
        _client(paths, tool_choice_mode="unknown-mode")


def test_codex_agent_thread_limit_is_validated_and_forwarded(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    with pytest.raises(ValueError, match="max_agent_threads must be in 1..32"):
        _client(paths, max_agent_threads=0)

    _set_fake(paths, response={"schema_version": 1, "patch": "diff\n"})
    client = _client(paths, max_agent_threads=2)
    asyncio.run(client.generate("bounded agents"))
    invocation = json.loads(paths["log"].read_text(encoding="utf-8"))

    args = invocation["args"]
    assert "--enable" in args
    assert args[args.index("--enable") + 1] == "multi_agent"
    disabled_features = {
        args[index + 1]
        for index, value in enumerate(args[:-1])
        if value == "--disable"
    }
    assert "multi_agent_v2" in disabled_features
    assert "agents.enabled=true" in args
    assert "agents.max_concurrent_threads_per_session=2" in args


def test_codex_model_catalog_is_private_and_forwarded(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(paths, response={"schema_version": 1, "patch": "diff\n"})

    asyncio.run(
        _client(
            paths,
            model_catalog_path=str(paths["model_catalog"]),
        ).generate("catalog")
    )
    invocation = json.loads(paths["log"].read_text(encoding="utf-8"))
    args = invocation["args"]

    catalog_override = next(
        value for value in args if value.startswith("model_catalog_json=")
    )
    private_path = Path(json.loads(catalog_override.split("=", 1)[1]))
    assert private_path.name == "model_catalog.json"
    assert private_path.parent == Path(invocation["home"])
    assert str(paths["model_catalog"]) not in args
    assert invocation["model_catalog_exists"] is True
    assert invocation["model_catalog_mode"] == 0o600
    assert json.loads(invocation["model_catalog_text"]) == {
        "models": [{"slug": "gpt-5.6-sol"}]
    }
    assert not private_path.exists()


def test_required_first_tool_choice_needs_safe_explicit_base_url(
    tmp_path: Path,
) -> None:
    paths = _make_inputs(tmp_path)

    with pytest.raises(ValueError, match="safe explicit provider base_url"):
        _client(paths, tool_choice_mode="required-first")

    paths["config"].write_text(
        'model_provider = "OpenAI"\n'
        'model = "gpt-5.6-sol"\n'
        'model_reasoning_effort = "ultra"\n'
        '[model_providers.OpenAI]\n'
        'name = "OpenAI"\n'
        'base_url = "http://provider.example/v1"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="absolute HTTPS URL"):
        _client(paths, tool_choice_mode="required-first")


def test_required_first_tool_choice_uses_private_per_attempt_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_inputs(tmp_path)
    paths["config"].write_text(
        'model_provider = "kimi"\n'
        'model = "k3"\n'
        'model_reasoning_effort = "ultra"\n'
        '[model_providers.kimi]\n'
        'name = "kimi"\n'
        'base_url = "https://provider.example/v1"\n'
        'wire_api = "responses"\n',
        encoding="utf-8",
    )
    _set_fake(paths, response={"schema_version": 1, "patch": "diff\n"})
    captured: dict[str, object] = {}

    class FakeProxy:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.base_url = None
            self.client_api_key = kwargs["client_api_key"]

        async def __aenter__(self):
            self.base_url = "http://127.0.0.1:43210/v1"
            return self

        async def __aexit__(self, *_args):
            captured["closed"] = True
            self.base_url = None

    import simpletes.llm.codex_exec as codex_exec

    monkeypatch.setattr(codex_exec, "RequiredFirstToolProxy", FakeProxy)

    asyncio.run(
        _client(
            paths,
            model="k3",
            tool_choice_mode="required-first",
        ).preflight("inspect repo")
    )
    record = json.loads(paths["log"].read_text(encoding="utf-8"))
    args = record["args"]

    proxy_kwargs = captured["kwargs"]
    assert proxy_kwargs["upstream_base_url"] == "https://provider.example/v1"
    assert proxy_kwargs["upstream_api_key"] == "unit-test-secret"
    assert proxy_kwargs["client_api_key"] != "unit-test-secret"
    assert proxy_kwargs["timeout"] == 10
    assert captured["closed"] is True
    assert 'model_providers.kimi.base_url="http://127.0.0.1:43210/v1"' in args
    assert "https://provider.example/v1" not in json.dumps(args)
    assert "unit-test-secret" not in json.dumps(args)
    assert record["inherited_api_key"] == proxy_kwargs["client_api_key"]
    disabled_features = {
        args[index + 1]
        for index, value in enumerate(args[:-1])
        if value == "--disable"
    }
    assert disabled_features == {"plugins", "enable_request_compression"}


def test_required_first_concurrent_attempts_use_isolated_redaction_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_inputs(tmp_path)
    paths["config"].write_text(
        'model_provider = "kimi"\n'
        'model = "k3"\n'
        'model_reasoning_effort = "ultra"\n'
        '[model_providers.kimi]\n'
        'name = "kimi"\n'
        'base_url = "https://provider.example/v1"\n'
        'wire_api = "responses"\n',
        encoding="utf-8",
    )
    captured_keys: list[str] = []

    class FakeProxy:
        def __init__(self, **kwargs):
            self.client_api_key = kwargs["client_api_key"]
            self.base_url = None
            captured_keys.append(self.client_api_key)

        async def __aenter__(self):
            self.base_url = f"http://127.0.0.1:{43000 + len(captured_keys)}/v1"
            return self

        async def __aexit__(self, *_args):
            self.base_url = None

    import simpletes.llm.codex_exec as codex_exec

    monkeypatch.setattr(codex_exec, "RequiredFirstToolProxy", FakeProxy)
    client = _client(
        paths,
        model="k3",
        tool_choice_mode="required-first",
    )

    async def collect_overrides():
        async def collect_one():
            async with client._provider_override() as provider:
                await asyncio.sleep(0)
                return provider

        return await asyncio.gather(collect_one(), collect_one())

    providers = asyncio.run(collect_overrides())

    assert len(captured_keys) == 2
    assert len(set(captured_keys)) == 2
    assert client._auth_secrets == frozenset({"unit-test-secret"})
    for provider in providers:
        assert provider.temporary_secrets == frozenset({provider.process_api_key})
        assert client._safe_diagnostic(
            provider.process_api_key.encode("utf-8"),
            temporary_secrets=provider.temporary_secrets,
        ) == "[REDACTED]"
        with pytest.raises(LLMCallError) as raised:
            client._parse_response(
                json.dumps(
                    {
                        "schema_version": 1,
                        "patch": provider.process_api_key,
                    }
                ),
                temporary_secrets=provider.temporary_secrets,
            )
        assert raised.value.error_type == "SensitiveOutput"
        assert provider.process_api_key not in str(raised.value)


def test_cli_builds_local_validation_overlay_config(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    args = build_parser().parse_args(
        [
            "--init-program",
            str(tmp_path / "init.py"),
            "--evaluator",
            str(tmp_path / "evaluator.py"),
            "--instruction",
            str(tmp_path / "instruction.txt"),
            "--instruction-suffix",
            str(tmp_path / "instruction.k3.txt"),
            "--llm-backend",
            "codex_exec",
            "--codex-config",
            str(paths["config"]),
            "--codex-auth",
            str(paths["auth"]),
            "--codex-repo-root",
            str(paths["repo"]),
            "--codex-output-schema",
            str(paths["schema"]),
            "--codex-local-validation-schema",
            str(paths["local_schema"]),
            "--codex-output-mode",
            "local-json",
            "--codex-tool-choice-mode",
            "required-first",
            "--codex-max-agent-threads",
            "2",
            "--codex-model-catalog",
            str(paths["model_catalog"]),
        ]
    )

    config = build_config_from_args(args)

    assert config.codex_local_validation_schema == str(paths["local_schema"])
    assert config.instruction_suffix_path == str(tmp_path / "instruction.k3.txt")
    assert config.codex_output_mode == "local-json"
    assert config.codex_tool_choice_mode == "required-first"
    assert config.codex_max_agent_threads == 2
    assert config.codex_model_catalog_path == str(paths["model_catalog"])


def test_factory_passes_overlay_and_repairs_empty_provider_response(
    tmp_path: Path,
) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake_sequence(
        paths,
        [
            {"response": {"schema_version": 1, "patch": ""}},
            {"response": {"schema_version": 1, "patch": "factory diff\n"}},
        ],
    )
    config = EngineConfig(
        init_program=str(tmp_path / "init.py"),
        evaluator_path=str(tmp_path / "evaluator.py"),
        instruction_path=str(tmp_path / "instruction.txt"),
        llm_backend="codex_exec",
        model="gpt-5.6-sol",
        reasoning_effort="ultra",
        codex_config_path=str(paths["config"]),
        codex_auth_path=str(paths["auth"]),
        codex_repo_root=str(paths["repo"]),
        codex_output_schema=str(paths["schema"]),
        codex_local_validation_schema=str(paths["local_schema"]),
        codex_output_mode="local-json",
        codex_tool_choice_mode="auto",
        codex_model_catalog_path=str(paths["model_catalog"]),
        timeout=10,
    )

    client = create_llm_client(config)
    assert isinstance(client, CodexExecClient)
    client.codex_binary = str(paths["executable"])
    result = asyncio.run(client.generate("factory prompt", track_io=True))
    tracked = json.loads(result.raw_output or "")

    assert tracked["response"]["patch"] == "factory diff\n"
    assert "SemanticValidationError" in tracked["simpletes_codex_audit"][
        "failure_summaries"
    ][0]
    invocation = json.loads(paths["log"].read_text(encoding="utf-8"))
    assert "--output-schema" not in invocation["args"]


def test_custom_provider_receives_credential_override(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    paths["config"].write_text(
        'model_provider = "kimi"\n'
        'model = "k3"\n'
        'model_reasoning_effort = "ultra"\n'
        '[model_providers.kimi]\n'
        'name = "kimi"\n'
        'base_url = "https://provider.example"\n'
        'wire_api = "responses"\n'
        'requires_openai_auth = true\n',
        encoding="utf-8",
    )
    _set_fake(paths, response={"schema_version": 1, "patch": ""})

    asyncio.run(_client(paths, model="k3").preflight())
    record = json.loads(paths["log"].read_text(encoding="utf-8"))
    args = record["args"]

    assert args[args.index("-m") + 1] == "k3"
    assert "model_providers.kimi.requires_openai_auth=false" in args
    assert 'model_providers.kimi.env_key="OPENAI_API_KEY"' in args
    assert "model_providers.OpenAI.requires_openai_auth=false" not in args
    assert not record["auth_exists"]
    assert record["inherited_api_key"] == "unit-test-secret"
    assert str(paths["auth"]) not in args
    assert "unit-test-secret" not in json.dumps(args)


def test_custom_provider_contract_fails_closed(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    paths["config"].write_text(
        'model_provider = "kimi"\n'
        'model = "gpt-5.6-sol"\n'
        'model_reasoning_effort = "ultra"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no matching model_providers.kimi table"):
        _client(paths)

    paths["config"].write_text(
        'model_provider = "bad.provider"\n'
        'model = "gpt-5.6-sol"\n'
        'model_reasoning_effort = "ultra"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bare-key-compatible"):
        _client(paths)


def test_legacy_config_without_provider_uses_openai_fallback(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    paths["config"].write_text(
        'model = "gpt-5.6-sol"\nmodel_reasoning_effort = "ultra"\n',
        encoding="utf-8",
    )
    _set_fake(paths, response={"schema_version": 1, "patch": ""})

    asyncio.run(_client(paths).preflight())
    record = json.loads(paths["log"].read_text(encoding="utf-8"))
    assert "model_providers.OpenAI.requires_openai_auth=false" in record["args"]


def test_auth_symbolic_link_is_rejected_before_resolution(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    auth_link = tmp_path / "auth-link.json"
    auth_link.symlink_to(paths["auth"])

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        _client(paths, auth_path=str(auth_link))


def test_generate_batch_and_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(paths, response={"schema_version": 1, "patch": ""})
    client = _client(paths)

    asyncio.run(client.preflight())
    results = asyncio.run(client.generate_batch("prompt", n=2))
    assert len(results) == 2
    assert all("# EVOLVE-BLOCK-START" in result.text for result in results)
    assert asyncio.run(client.generate_batch("prompt", n=0)) == []


def test_generate_discards_response_that_echoes_auth_secret(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(
        paths,
        response={"schema_version": 1, "patch": "unit-test-secret"},
    )

    with pytest.raises(LLMCallError) as raised:
        asyncio.run(_client(paths).generate("prompt", track_io=True))
    assert raised.value.error_type == "SensitiveOutput"
    assert "unit-test-secret" not in str(raised.value)


def test_generate_discards_json_escaped_auth_secret(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(
        paths,
        raw='{"schema_version":1,"patch":"unit-test-\\u0073ecret"}',
    )

    with pytest.raises(LLMCallError) as raised:
        asyncio.run(_client(paths).generate("prompt", track_io=True))
    assert raised.value.error_type == "SensitiveOutput"
    assert "unit-test-secret" not in str(raised.value)
