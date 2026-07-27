from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

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
            }
        ),
        encoding="utf-8",
    )
    executable.write_text(
        """#!/usr/bin/env python3
import json
import os
import pathlib
import stat
import sys

args = sys.argv[1:]
base = pathlib.Path(__file__).resolve().parent
settings = json.loads((base / "fake-settings.json").read_text())
home = pathlib.Path(os.environ["CODEX_HOME"])
record = {
    "args": args,
    "home": str(home),
    "home_mode": stat.S_IMODE(home.stat().st_mode),
    "config_mode": stat.S_IMODE((home / "config.toml").stat().st_mode),
    "auth_exists": (home / "auth.json").exists(),
    "inherited_api_key": os.environ.get("OPENAI_API_KEY"),
    "inherited_webhook": os.environ.get("WX_WEBHOOK_URL"),
    "inherited_github_token": os.environ.get("GITHUB_TOKEN"),
    "inherited_arbitrary": os.environ.get("ARBITRARY_PARENT_VALUE"),
    "prompt": sys.stdin.read(),
}
(base / "invocation.json").write_text(json.dumps(record))
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
        "repo": repo,
        "executable": executable,
        "settings": settings,
        "log": tmp_path / "invocation.json",
    }


def _set_fake(
    paths: dict[str, Path],
    *,
    response: object = None,
    raw: str | None = None,
    status: int = 0,
    stderr: str = "",
) -> None:
    paths["settings"].write_text(
        json.dumps(
            {"response": response, "raw": raw, "stderr": stderr, "status": status}
        ),
        encoding="utf-8",
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
    assert "--ignore-rules" in args
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


def test_generate_does_not_echo_failed_subprocess_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_inputs(tmp_path)
    _set_fake(
        paths,
        response={},
        status=7,
        stderr="provider accidentally printed unit-test-secret",
    )

    with pytest.raises(LLMCallError) as raised:
        asyncio.run(_client(paths).generate("prompt"))
    assert raised.value.error_type == "CodexExecError"
    assert "unit-test-secret" not in str(raised.value)


def test_config_model_and_effort_must_match(tmp_path: Path) -> None:
    paths = _make_inputs(tmp_path)
    with pytest.raises(ValueError, match="model mismatch"):
        _client(paths, model="different-model")
    with pytest.raises(ValueError, match="reasoning-effort mismatch"):
        _client(paths, reasoning_effort="high")


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
