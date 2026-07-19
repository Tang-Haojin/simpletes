"""Secure Codex CLI backend for repository-aware structured generation.

The backend deliberately delegates repository inspection to ``codex exec``
instead of exposing credentials to SimpleTES evaluators.  Every request gets a
fresh, private ``CODEX_HOME`` containing only the selected configuration.  The
API key is read from the selected auth JSON and passed only to the parent Codex
process; generated tool shells inherit no parent environment.  The directory
is removed as soon as the request finishes.
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import stat
import tempfile
import tomllib
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from simpletes.llm.types import LLMCallError, LLMResult


_START_MARKER = "# EVOLVE-BLOCK-START"
_END_MARKER = "# EVOLVE-BLOCK-END"
_SAFE_ENVIRONMENT_KEYS = frozenset(
    {
        "COLORTERM",
        "HOME",
        "LANG",
        "LANGUAGE",
        "LOGNAME",
        "PATH",
        "SHELL",
        "TERM",
        "TERM_PROGRAM",
        "TERM_PROGRAM_VERSION",
        "TMP",
        "TMPDIR",
        "TEMP",
        "TZ",
        "USER",
    }
)
_SAFE_ENVIRONMENT_PREFIXES = ("LC_",)
_SENSITIVE_DIAGNOSTIC_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s\"']+"),
    re.compile(
        r"(?i)((?:api[_-]?key|auth[_-]?token|access[_-]?token|password|secret)"
        r"\s*[=:]\s*[\"']?)[^\s,\"']+"
    ),
)


class CodexExecClient:
    """Run Codex non-interactively with a JSON-schema-constrained response."""

    def __init__(
        self,
        *,
        model: str,
        reasoning_effort: str,
        config_path: str,
        auth_path: str,
        repo_root: str,
        output_schema: str,
        timeout: float | None = None,
        pool_size: int = 1,
        codex_binary: str = "codex",
    ) -> None:
        del pool_size  # The engine owns generation concurrency.
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.config_path = self._resolve_regular_input(Path(config_path), "Codex config")
        self.auth_path = self._resolve_regular_input(Path(auth_path), "Codex auth")
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.output_schema_path = self._resolve_regular_input(
            Path(output_schema), "Codex output schema"
        )
        self.timeout = timeout
        self.codex_binary = codex_binary

        self._validate_inputs()

    def _error(self, error_type: str, message: str) -> LLMCallError:
        return LLMCallError(
            model=self.model,
            api_base=None,
            error_type=error_type,
            message=message,
        )

    @staticmethod
    def _require_regular_file(path: Path, label: str) -> None:
        try:
            mode = path.stat().st_mode
        except OSError as exc:
            raise ValueError(f"{label} is not readable: {path}") from exc
        if not stat.S_ISREG(mode):
            raise ValueError(f"{label} must be a regular file: {path}")

    @classmethod
    def _resolve_regular_input(cls, path: Path, label: str) -> Path:
        expanded = path.expanduser()
        if expanded.is_symlink():
            raise ValueError(f"{label} must not be a symbolic link: {expanded}")
        resolved = expanded.resolve()
        cls._require_regular_file(resolved, label)
        return resolved

    def _validate_inputs(self) -> None:
        self._require_regular_file(self.config_path, "Codex config")
        self._require_regular_file(self.auth_path, "Codex auth")
        self._require_regular_file(self.output_schema_path, "Codex output schema")
        if not self.repo_root.is_dir():
            raise ValueError(f"Codex repository root is not a directory: {self.repo_root}")

        try:
            with self.config_path.open("rb") as stream:
                config = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError("Codex config is not valid TOML") from exc

        configured_model = config.get("model")
        configured_effort = config.get("model_reasoning_effort")
        if configured_model != self.model:
            raise ValueError(
                "Codex model mismatch between SimpleTES and the selected config "
                f"({self.model!r} != {configured_model!r})"
            )
        if configured_effort != self.reasoning_effort:
            raise ValueError(
                "Codex reasoning-effort mismatch between SimpleTES and the selected config "
                f"({self.reasoning_effort!r} != {configured_effort!r})"
            )

        try:
            with self.auth_path.open(encoding="utf-8") as stream:
                auth = json.load(stream)
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Codex auth is not valid JSON") from exc
        if not isinstance(auth, dict) or not auth:
            raise ValueError("Codex auth must be a non-empty JSON object")
        api_key = auth.get("OPENAI_API_KEY")
        if not isinstance(api_key, str) or not api_key:
            raise ValueError("Codex auth must contain a non-empty OPENAI_API_KEY")
        self._api_key = api_key
        self._auth_secrets = frozenset(self._collect_secret_values(auth))
        if not self._auth_secrets:
            raise ValueError("Codex auth must contain at least one non-empty secret value")

        try:
            with self.output_schema_path.open(encoding="utf-8") as stream:
                schema = json.load(stream)
            Draft202012Validator.check_schema(schema)
        except (OSError, UnicodeError, json.JSONDecodeError, SchemaError) as exc:
            raise ValueError("Codex output schema is not a valid JSON Schema") from exc
        self._schema: dict[str, Any] = schema
        self._validator = Draft202012Validator(schema)

    @classmethod
    def _collect_secret_values(cls, value: Any) -> set[str]:
        """Collect auth scalar values without ever placing them in diagnostics."""
        if isinstance(value, dict):
            result: set[str] = set()
            for nested in value.values():
                result.update(cls._collect_secret_values(nested))
            return result
        if isinstance(value, list):
            result = set()
            for nested in value:
                result.update(cls._collect_secret_values(nested))
            return result
        if isinstance(value, str) and value:
            return {value}
        return set()

    @staticmethod
    def _safe_environment(codex_home: Path) -> dict[str, str]:
        """Build a minimal allowlisted environment for Codex and its tools."""
        safe = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in _SAFE_ENVIRONMENT_KEYS
            or key.upper().startswith(_SAFE_ENVIRONMENT_PREFIXES)
        }
        # Prevent tools from resolving the user's actual home.  Codex itself
        # uses CODEX_HOME, and tool-shell inheritance is disabled separately.
        safe["HOME"] = str(codex_home)
        safe["CODEX_HOME"] = str(codex_home)
        return safe

    @staticmethod
    def _copy_private(source: Path, destination: Path) -> None:
        shutil.copyfile(source, destination)
        destination.chmod(0o600)

    def _parse_response(self, response_text: str) -> tuple[dict[str, Any], str]:
        if any(secret in response_text for secret in self._auth_secrets):
            raise self._error(
                "SensitiveOutput",
                "Codex response contained authentication material and was discarded",
            )
        try:
            value = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise self._error("InvalidJSON", "Codex returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise self._error("SchemaValidationError", "Codex response must be a JSON object")
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
        if any(secret in canonical for secret in self._auth_secrets):
            raise self._error(
                "SensitiveOutput",
                "Codex response contained authentication material and was discarded",
            )
        try:
            self._validator.validate(value)
        except ValidationError as exc:
            location = "/".join(str(part) for part in exc.absolute_path) or "<root>"
            raise self._error(
                "SchemaValidationError",
                f"Codex response failed schema validation at {location}: {exc.message}",
            ) from exc

        marked = (
            "```python\n"
            f"{_START_MARKER}\n"
            f"{canonical}\n"
            f"{_END_MARKER}\n"
            "```"
        )
        return value, marked

    def _safe_diagnostic(self, raw: bytes | None) -> str:
        text = (raw or b"").decode("utf-8", errors="replace")
        for secret in sorted(self._auth_secrets, key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
        for pattern in _SENSITIVE_DIAGNOSTIC_PATTERNS:
            if pattern.groups:
                text = pattern.sub(lambda match: f"{match.group(1)}[REDACTED]", text)
            else:
                text = pattern.sub("[REDACTED]", text)
        compact = " ".join(text.strip().split())
        return compact[-2000:] if compact else "no diagnostic text"

    async def _invoke(self, prompt: str) -> tuple[str, str]:
        """Run one isolated Codex request and return marked and raw outputs."""
        if shutil.which(self.codex_binary) is None:
            raise self._error("CodexNotFound", "Codex CLI executable was not found")

        with tempfile.TemporaryDirectory(prefix="simpletes-codex-") as tmp_name:
            codex_home = Path(tmp_name)
            codex_home.chmod(0o700)
            self._copy_private(self.config_path, codex_home / "config.toml")
            output_path = codex_home / "last-message.json"

            process_environment = self._safe_environment(codex_home)
            process_environment["OPENAI_API_KEY"] = self._api_key
            safe_path = process_environment.get("PATH", "/usr/bin:/bin")

            command = [
                self.codex_binary,
                "-a",
                "never",
                "exec",
                "--ephemeral",
                "--ignore-rules",
                "--sandbox",
                "read-only",
                "-C",
                str(self.repo_root),
                "-m",
                self.model,
                "-c",
                f'model_reasoning_effort="{self.reasoning_effort}"',
                "-c",
                "model_providers.OpenAI.requires_openai_auth=false",
                "-c",
                'model_providers.OpenAI.env_key="OPENAI_API_KEY"',
                "-c",
                "disable_response_storage=true",
                "-c",
                'network_access="disabled"',
                "-c",
                'shell_environment_policy.inherit="none"',
                "-c",
                f"shell_environment_policy.set.PATH={json.dumps(safe_path)}",
                "-c",
                f"shell_environment_policy.set.HOME={json.dumps(str(codex_home))}",
                "-c",
                "mcp_servers={}",
                "-c",
                "notify=[]",
                "--output-schema",
                str(self.output_schema_path),
                "--color",
                "never",
                "-o",
                str(output_path),
                "-",
            ]
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.PIPE,
                    env=process_environment,
                    start_new_session=True,
                )
                communicate = process.communicate(prompt.encode("utf-8"))
                if self.timeout is None:
                    _, stderr = await communicate
                else:
                    _, stderr = await asyncio.wait_for(communicate, timeout=self.timeout)
            except (TimeoutError, asyncio.CancelledError) as exc:
                if "process" in locals() and process.returncode is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    await process.wait()
                if isinstance(exc, asyncio.CancelledError):
                    raise
                raise self._error("TimeoutError", "Codex CLI request timed out") from exc
            except OSError as exc:
                raise self._error("CodexLaunchError", "Codex CLI could not be launched") from exc

            if process.returncode != 0:
                diagnostic = self._safe_diagnostic(stderr)
                raise self._error(
                    "CodexExecError",
                    f"Codex CLI exited with status {process.returncode}: {diagnostic}",
                )
            try:
                response_text = output_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise self._error("MissingOutput", "Codex CLI produced no readable final response") from exc
            _, marked = self._parse_response(response_text)
            canonical = json.dumps(
                json.loads(response_text), ensure_ascii=False, sort_keys=True, indent=2
            )
            return marked, canonical

    async def preflight(self) -> None:
        """Exercise the configured model, credentials, and response schema."""
        await self._invoke(
            "Return one harmless placeholder JSON object that conforms exactly to "
            "the supplied output schema. Do not modify the repository."
        )

    async def generate(
        self,
        prompt: str,
        instance_id: str = "",
        track_io: bool = False,
    ) -> LLMResult:
        del instance_id
        marked, canonical = await self._invoke(prompt)
        return LLMResult(
            text=marked,
            prompt=prompt if track_io else None,
            raw_output=canonical if track_io else None,
            token_usage=None,
        )

    async def generate_batch(
        self,
        prompt: str,
        n: int,
        instance_id: str = "",
        track_io: bool = False,
    ) -> list[LLMResult]:
        if n <= 0:
            return []
        return list(
            await asyncio.gather(
                *(
                    self.generate(
                        prompt,
                        instance_id=f"{instance_id}-{index}",
                        track_io=track_io,
                    )
                    for index in range(n)
                )
            )
        )

    def close(self) -> None:
        """No persistent process or credential directory is retained."""
        return None


__all__ = ["CodexExecClient"]
