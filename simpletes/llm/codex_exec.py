"""Secure Codex CLI backend for repository-aware structured generation.

The backend deliberately delegates repository inspection to ``codex exec``
instead of exposing credentials to SimpleTES evaluators.  Every request gets a
fresh, private ``CODEX_HOME`` containing only the selected configuration.  The
API key is read from the selected auth JSON.  Normal requests pass it only to
the parent Codex process; required-first compatibility requests retain it in a
loopback proxy and give Codex an unrelated proxy credential.  Generated tool
shells inherit no parent environment.  The directory is removed as soon as the
request finishes.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import os
import re
import secrets
import shutil
import signal
import stat
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from simpletes.llm.responses_proxy import (
    RequiredFirstToolProxy,
    ResponsesProxyError,
    validate_upstream_base_url,
)
from simpletes.llm.types import LLMCallError, LLMResult


_START_MARKER = "# EVOLVE-BLOCK-START"
_END_MARKER = "# EVOLVE-BLOCK-END"
_PROVIDER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
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
_REPAIRABLE_ERROR_TYPES = frozenset(
    {"InvalidJSON", "SchemaValidationError", "SemanticValidationError"}
)
_MAX_REPAIR_ATTEMPTS = 2
_CLEANUP_RETRY_DELAYS = (0.0, 0.02, 0.05, 0.1, 0.2)
_MAX_CODEX_STDOUT_BYTES = 8 * 1024 * 1024
_MAX_CODEX_STDERR_BYTES = 1024 * 1024
_MAX_FINAL_OUTPUT_BYTES = 4 * 1024 * 1024
_OUTPUT_MODES = frozenset({"provider-structured", "local-json"})
_DEFAULT_OUTPUT_MODE = "provider-structured"
_TOOL_CHOICE_MODES = frozenset({"auto", "required-first"})
_DEFAULT_TOOL_CHOICE_MODE = "auto"
_REPO_TOOL_ITEM_TYPES = frozenset(
    {"command_execution", "exec_command", "file_read", "shell_command"}
)


@dataclass(frozen=True)
class CodexTraceSummary:
    """Bounded, secret-scrubbed audit data from one logical Codex request."""

    attempt_count: int
    event_count: int
    repo_tool_call_count: int
    repo_tool_types: tuple[str, ...]
    failure_summaries: tuple[str, ...]
    diagnostic_summaries: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_count": self.attempt_count,
            "event_count": self.event_count,
            "repo_tool_call_count": self.repo_tool_call_count,
            "repo_tool_types": list(self.repo_tool_types),
            "failure_summaries": list(self.failure_summaries),
            "diagnostic_summaries": list(self.diagnostic_summaries),
        }


@dataclass(frozen=True)
class CodexProbeResult:
    """Canonical structured result plus repository-inspection evidence."""

    value: dict[str, Any]
    canonical: str
    trace: CodexTraceSummary


@dataclass(frozen=True)
class _AttemptTrace:
    event_count: int
    repo_tool_call_count: int
    repo_tool_types: tuple[str, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class _ProviderOverride:
    base_url: str | None
    process_api_key: str
    temporary_secrets: frozenset[str]


ResponseValidator = Callable[[dict[str, Any], CodexTraceSummary], str | None]


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
        max_repair_attempts: int = _MAX_REPAIR_ATTEMPTS,
        response_validator: ResponseValidator | None = None,
        local_validation_schema: str | None = None,
        output_mode: str = _DEFAULT_OUTPUT_MODE,
        tool_choice_mode: str = _DEFAULT_TOOL_CHOICE_MODE,
    ) -> None:
        del pool_size  # The engine owns generation concurrency.
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.config_path = self._resolve_regular_input(
            Path(config_path), "Codex config"
        )
        self.auth_path = self._resolve_regular_input(Path(auth_path), "Codex auth")
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.output_schema_path = self._resolve_regular_input(
            Path(output_schema), "Codex output schema"
        )
        self.local_validation_schema_path = (
            self._resolve_regular_input(
                Path(local_validation_schema), "Codex local validation schema"
            )
            if local_validation_schema is not None
            else None
        )
        self.timeout = timeout
        self.codex_binary = codex_binary
        if not isinstance(output_mode, str) or output_mode not in _OUTPUT_MODES:
            choices = ", ".join(sorted(_OUTPUT_MODES))
            raise ValueError(f"Codex output_mode must be one of: {choices}")
        self.output_mode = output_mode
        if (
            not isinstance(tool_choice_mode, str)
            or tool_choice_mode not in _TOOL_CHOICE_MODES
        ):
            choices = ", ".join(sorted(_TOOL_CHOICE_MODES))
            raise ValueError(f"Codex tool_choice_mode must be one of: {choices}")
        self.tool_choice_mode = tool_choice_mode
        if (
            type(max_repair_attempts) is not int
            or not 0 <= max_repair_attempts <= _MAX_REPAIR_ATTEMPTS
        ):
            raise ValueError(
                f"Codex max_repair_attempts must be in 0..{_MAX_REPAIR_ATTEMPTS}"
            )
        self.max_repair_attempts = max_repair_attempts
        self._response_validator = response_validator

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
        if self.local_validation_schema_path is not None:
            self._require_regular_file(
                self.local_validation_schema_path,
                "Codex local validation schema",
            )
        if not self.repo_root.is_dir():
            raise ValueError(
                f"Codex repository root is not a directory: {self.repo_root}"
            )

        try:
            with self.config_path.open("rb") as stream:
                config = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError("Codex config is not valid TOML") from exc

        configured_model = config.get("model")
        configured_effort = config.get("model_reasoning_effort")
        configured_provider = config.get("model_provider", "OpenAI")
        if configured_model != self.model:
            raise ValueError(
                "Codex model mismatch between SimpleTES and the selected config "
                f"({self.model!r} != {configured_model!r})"
            )
        if configured_effort != self.reasoning_effort:
            raise ValueError(
                "Codex reasoning-effort mismatch between SimpleTES and the "
                "selected config "
                f"({self.reasoning_effort!r} != {configured_effort!r})"
            )
        if not isinstance(configured_provider, str) or not configured_provider:
            raise ValueError("Codex config model_provider must be a non-empty string")
        if _PROVIDER_NAME_PATTERN.fullmatch(configured_provider) is None:
            raise ValueError(
                "Codex config model_provider must be a TOML bare-key-compatible name"
            )
        providers = config.get("model_providers", {})
        if not isinstance(providers, dict):
            raise ValueError("Codex config model_providers must be a table")
        provider_config = providers.get(configured_provider)
        # OpenAI is a built-in Codex provider and remains the compatibility
        # fallback for older configs which omitted model_provider entirely.
        # Every named custom provider must have an explicit provider table so
        # the credential override cannot silently target the wrong endpoint.
        if provider_config is not None and not isinstance(provider_config, dict):
            raise ValueError(
                f"Codex config model_providers.{configured_provider} must be a table"
            )
        if configured_provider != "OpenAI" and provider_config is None:
            raise ValueError(
                "Codex config selected model_provider has no matching "
                f"model_providers.{configured_provider} table"
            )
        self._model_provider = configured_provider
        configured_base_url = (
            provider_config.get("base_url")
            if isinstance(provider_config, dict)
            else None
        )
        self._provider_base_url: str | None = None
        if self.tool_choice_mode == "required-first":
            try:
                self._provider_base_url = validate_upstream_base_url(
                    configured_base_url
                )
            except ResponsesProxyError as error:
                raise ValueError(
                    "Codex required-first tool choice needs a safe explicit "
                    f"provider base_url: {error}"
                ) from error

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
            raise ValueError(
                "Codex auth must contain at least one non-empty secret value"
            )

        try:
            with self.output_schema_path.open(encoding="utf-8") as stream:
                schema = json.load(stream)
            Draft202012Validator.check_schema(schema)
        except (OSError, UnicodeError, json.JSONDecodeError, SchemaError) as exc:
            raise ValueError("Codex output schema is not a valid JSON Schema") from exc
        self._schema: dict[str, Any] = schema
        self._validator = Draft202012Validator(schema)

        self._local_validator: Draft202012Validator | None = None
        if self.local_validation_schema_path is not None:
            try:
                with self.local_validation_schema_path.open(
                    encoding="utf-8"
                ) as stream:
                    local_schema = json.load(stream)
                Draft202012Validator.check_schema(local_schema)
            except (
                OSError,
                UnicodeError,
                json.JSONDecodeError,
                SchemaError,
            ) as exc:
                raise ValueError(
                    "Codex local validation schema is not a valid JSON Schema"
                ) from exc
            self._local_validator = Draft202012Validator(local_schema)

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

    def _parse_response(
        self,
        response_text: str,
        *,
        temporary_secrets: frozenset[str] = frozenset(),
    ) -> tuple[dict[str, Any], str]:
        secret_values = self._auth_secrets | temporary_secrets
        if any(secret in response_text for secret in secret_values):
            raise self._error(
                "SensitiveOutput",
                "Codex response contained authentication material and was discarded",
            )
        try:
            value = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise self._error("InvalidJSON", "Codex returned invalid JSON") from exc
        if not isinstance(value, dict):
            raise self._error(
                "SchemaValidationError", "Codex response must be a JSON object"
            )
        canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
        if any(secret in canonical for secret in secret_values):
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
        if self._local_validator is not None:
            try:
                self._local_validator.validate(value)
            except ValidationError as exc:
                location = "/".join(
                    str(part) for part in exc.absolute_path
                ) or "<root>"
                raise self._error(
                    "SemanticValidationError",
                    "Codex response failed local semantic validation at "
                    f"{location}: {exc.message}",
                ) from exc

        return value, canonical

    @staticmethod
    def _marked_response(canonical: str) -> str:
        return (
            "```python\n"
            f"{_START_MARKER}\n"
            f"{canonical}\n"
            f"{_END_MARKER}\n"
            "```"
        )

    def _safe_diagnostic(
        self,
        raw: bytes | None,
        *,
        temporary_secrets: frozenset[str] = frozenset(),
    ) -> str:
        text = (raw or b"").decode("utf-8", errors="replace")
        secret_values = self._auth_secrets | temporary_secrets
        for secret in sorted(secret_values, key=len, reverse=True):
            text = text.replace(secret, "[REDACTED]")
        for pattern in _SENSITIVE_DIAGNOSTIC_PATTERNS:
            if pattern.groups:
                text = pattern.sub(lambda match: f"{match.group(1)}[REDACTED]", text)
            else:
                text = pattern.sub("[REDACTED]", text)
        compact = " ".join(text.strip().split())
        return compact[-2000:] if compact else "no diagnostic text"

    @staticmethod
    def _read_bounded_file(
        path: Path, limit: int, *, tail: bool = False
    ) -> tuple[bytes, bool]:
        """Read at most ``limit`` bytes from a private subprocess log."""
        size = path.stat().st_size
        with path.open("rb") as stream:
            if tail and size > limit:
                stream.seek(-limit, os.SEEK_END)
            raw = stream.read(limit + 1)
        truncated = size > limit or len(raw) > limit
        return raw[:limit], truncated

    def _diagnostic_from_file(
        self,
        path: Path,
        *,
        temporary_secrets: frozenset[str] = frozenset(),
    ) -> str:
        try:
            raw, truncated = self._read_bounded_file(
                path, _MAX_CODEX_STDERR_BYTES, tail=True
            )
        except OSError:
            return "no diagnostic text"
        summary = self._safe_diagnostic(
            raw, temporary_secrets=temporary_secrets
        )
        if truncated and summary != "no diagnostic text":
            return f"stderr tail (truncated): {summary}"
        return summary

    def _summarize_event_stream(
        self,
        stdout_path: Path,
        stderr_path: Path,
        *,
        temporary_secrets: frozenset[str] = frozenset(),
    ) -> _AttemptTrace:
        """Reduce Codex JSONL to counts/types; never retain event payloads."""
        event_count = 0
        tool_ids: set[str] = set()
        tool_types: set[str] = set()
        diagnostics: list[str] = []
        try:
            stdout, stdout_truncated = self._read_bounded_file(
                stdout_path, _MAX_CODEX_STDOUT_BYTES
            )
        except OSError:
            stdout = b""
            stdout_truncated = False
            diagnostics.append("stdout JSONL log was unavailable")
        text = stdout.decode("utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if len(diagnostics) < 4:
                    diagnostics.append(f"stdout JSONL line {line_number} was invalid")
                continue
            if not isinstance(event, dict):
                if len(diagnostics) < 4:
                    diagnostics.append(
                        f"stdout JSONL line {line_number} was not an object"
                    )
                continue
            event_count += 1
            if event.get("type") not in {"item.started", "item.completed"}:
                continue
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if not isinstance(item_type, str) or item_type not in _REPO_TOOL_ITEM_TYPES:
                continue
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id:
                # A completed item without an id is still evidence. Started
                # records without ids are skipped to avoid double counting.
                if event.get("type") != "item.completed":
                    continue
                item_id = f"line-{line_number}"
            tool_ids.add(f"{item_type}:{item_id}")
            tool_types.add(item_type)

        if stdout_truncated:
            diagnostics.append(
                "stdout JSONL exceeded the bounded audit limit; parsed prefix only"
            )
        stderr_summary = self._diagnostic_from_file(
            stderr_path, temporary_secrets=temporary_secrets
        )
        if stderr_summary != "no diagnostic text":
            diagnostics.append(stderr_summary)
        return _AttemptTrace(
            event_count=event_count,
            repo_tool_call_count=len(tool_ids),
            repo_tool_types=tuple(sorted(tool_types)),
            diagnostics=tuple(diagnostics[:8]),
        )

    @staticmethod
    def _aggregate_trace(
        attempts: list[_AttemptTrace], failures: list[str]
    ) -> CodexTraceSummary:
        return CodexTraceSummary(
            attempt_count=len(attempts),
            event_count=sum(attempt.event_count for attempt in attempts),
            repo_tool_call_count=sum(
                attempt.repo_tool_call_count for attempt in attempts
            ),
            repo_tool_types=tuple(
                sorted(
                    {
                        item_type
                        for attempt in attempts
                        for item_type in attempt.repo_tool_types
                    }
                )
            ),
            failure_summaries=tuple(failures),
            diagnostic_summaries=tuple(
                diagnostic
                for attempt in attempts
                for diagnostic in attempt.diagnostics
            )[:16],
        )

    def _failure_summary(self, attempt: int, error: LLMCallError) -> str:
        safe_message = self._safe_diagnostic(error.message.encode("utf-8"))
        return f"attempt {attempt}: {error.error_type}: {safe_message}"[-1200:]

    def _repair_prompt(
        self,
        original_prompt: str,
        response_text: str,
        error: LLMCallError,
        attempt: int,
    ) -> str:
        safe_error = self._safe_diagnostic(error.message.encode("utf-8"))
        safe_response = self._safe_diagnostic(response_text.encode("utf-8"))
        return (
            f"{original_prompt.rstrip()}\n\n"
            "MANDATORY STRUCTURED-RESPONSE REPAIR\n"
            f"The final response from attempt {attempt} was rejected.\n"
            f"Exact validation error: {safe_error}\n"
            f"Rejected final response (sanitized): {safe_response}\n"
            "Re-do any repository inspection needed to satisfy the original request. "
            "Return one corrected JSON object conforming exactly to the response "
            "contract in the original request. Do not return prose, Markdown fences, "
            "a placeholder, or a control candidate."
        )

    async def _remove_private_home(self, codex_home: Path) -> None:
        """Remove an ephemeral home, tolerating a short-lived writer race."""
        last_error: OSError | None = None
        for delay in _CLEANUP_RETRY_DELAYS:
            if delay:
                await asyncio.sleep(delay)
            try:
                shutil.rmtree(codex_home)
                return
            except FileNotFoundError:
                return
            except OSError as error:
                last_error = error
        raise self._error(
            "CodexCleanupError",
            "Codex isolated home could not be removed after bounded retries",
        ) from last_error

    @asynccontextmanager
    async def _private_home(self) -> AsyncIterator[Path]:
        codex_home = Path(tempfile.mkdtemp(prefix="simpletes-codex-"))
        codex_home.chmod(0o700)
        try:
            yield codex_home
        finally:
            await self._remove_private_home(codex_home)

    @asynccontextmanager
    async def _provider_override(self) -> AsyncIterator[_ProviderOverride]:
        """Yield a provider URL override and the credential given to Codex."""
        if self.tool_choice_mode == "auto":
            yield _ProviderOverride(
                base_url=None,
                process_api_key=self._api_key,
                temporary_secrets=frozenset(),
            )
            return
        if self._provider_base_url is None:
            raise self._error(
                "CodexProxyError",
                "Codex required-first tool proxy is not fully configured",
            )
        client_api_key = secrets.token_urlsafe(32)
        temporary_secrets = frozenset({client_api_key})
        try:
            async with RequiredFirstToolProxy(
                upstream_base_url=self._provider_base_url,
                upstream_api_key=self._api_key,
                client_api_key=client_api_key,
                timeout=self.timeout,
            ) as proxy:
                if proxy.base_url is None:
                    raise ResponsesProxyError("loopback proxy has no bound base URL")
                yield _ProviderOverride(
                    base_url=proxy.base_url,
                    process_api_key=proxy.client_api_key,
                    temporary_secrets=temporary_secrets,
                )
        except ResponsesProxyError as error:
            raise self._error(
                "CodexProxyError",
                self._safe_diagnostic(
                    str(error).encode("utf-8"),
                    temporary_secrets=temporary_secrets,
                ),
            ) from error

    @staticmethod
    async def _kill_process(process: asyncio.subprocess.Process) -> None:
        """Kill a Codex process group, falling back to the direct child."""
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        await process.wait()

    def _read_final_output(self, output_path: Path) -> str:
        try:
            mode = output_path.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise OSError("Codex final response is not a regular file")
            if output_path.stat().st_size > _MAX_FINAL_OUTPUT_BYTES:
                raise self._error(
                    "OutputTooLarge",
                    "Codex final response exceeded the bounded output limit",
                )
            with output_path.open("rb") as stream:
                raw = stream.read(_MAX_FINAL_OUTPUT_BYTES + 1)
            if len(raw) > _MAX_FINAL_OUTPUT_BYTES:
                raise self._error(
                    "OutputTooLarge",
                    "Codex final response exceeded the bounded output limit",
                )
            return raw.decode("utf-8")
        except LLMCallError:
            raise
        except (OSError, UnicodeError) as exc:
            raise self._error(
                "MissingOutput",
                "Codex CLI produced no readable final response",
            ) from exc

    async def _invoke_once(
        self, prompt: str
    ) -> tuple[str, _AttemptTrace, frozenset[str]]:
        """Run one isolated Codex attempt and return its final response and trace."""
        if shutil.which(self.codex_binary) is None:
            raise self._error("CodexNotFound", "Codex CLI executable was not found")

        async with (
            self._private_home() as codex_home,
            self._provider_override() as provider,
        ):
            self._copy_private(self.config_path, codex_home / "config.toml")
            output_path = codex_home / "last-message.json"
            stdout_path = codex_home / "events.jsonl"
            stderr_path = codex_home / "stderr.log"

            process_environment = self._safe_environment(codex_home)
            process_environment["OPENAI_API_KEY"] = provider.process_api_key
            safe_path = process_environment.get("PATH", "/usr/bin:/bin")
            provider_prefix = f"model_providers.{self._model_provider}"

            command = [
                self.codex_binary,
                "-a",
                "never",
                "exec",
                "--ephemeral",
                "--disable",
                "plugins",
            ]
            # The loopback compatibility proxy deliberately accepts plain JSON
            # only.  Preserve Codex's normal compression behavior for every
            # direct provider path and disable it only while that proxy is in
            # use.
            if self.tool_choice_mode == "required-first":
                command.extend(["--disable", "enable_request_compression"])
            command.extend(
                [
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
                f"{provider_prefix}.requires_openai_auth=false",
                "-c",
                f'{provider_prefix}.env_key="OPENAI_API_KEY"',
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
                "--json",
                ]
            )
            if provider.base_url is not None:
                command.extend(
                    [
                        "-c",
                        f"{provider_prefix}.base_url={json.dumps(provider.base_url)}",
                    ]
                )
            if self.output_mode == "provider-structured":
                command.extend(
                    ["--output-schema", str(self.output_schema_path)]
                )
            command.extend(
                [
                    "--color",
                    "never",
                    "-o",
                    str(output_path),
                    "-",
                ]
            )
            process: asyncio.subprocess.Process | None = None
            with stdout_path.open("wb") as stdout_stream, stderr_path.open(
                "wb"
            ) as stderr_stream:
                stdout_path.chmod(0o600)
                stderr_path.chmod(0o600)
                try:
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=stdout_stream,
                        stderr=stderr_stream,
                        env=process_environment,
                        start_new_session=True,
                    )
                    communicate = process.communicate(prompt.encode("utf-8"))
                    if self.timeout is None:
                        await communicate
                    else:
                        await asyncio.wait_for(communicate, timeout=self.timeout)
                except (TimeoutError, asyncio.CancelledError) as exc:
                    if process is not None and process.returncode is None:
                        await self._kill_process(process)
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                    timeout_trace = self._summarize_event_stream(
                        stdout_path,
                        stderr_path,
                        temporary_secrets=provider.temporary_secrets,
                    )
                    trace_json = json.dumps(
                        self._aggregate_trace([timeout_trace], []).to_dict(),
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                    raise self._error(
                        "TimeoutError",
                        "Codex CLI request timed out; "
                        f"sanitized_trace={trace_json}",
                    ) from exc
                except OSError as exc:
                    if process is not None and process.returncode is None:
                        await self._kill_process(process)
                    raise self._error(
                        "CodexLaunchError", "Codex CLI could not be launched"
                    ) from exc

            assert process is not None
            if process.returncode != 0:
                diagnostic = self._diagnostic_from_file(
                    stderr_path,
                    temporary_secrets=provider.temporary_secrets,
                )
                raise self._error(
                    "CodexExecError",
                    f"Codex CLI exited with status {process.returncode}: {diagnostic}",
                )
            response_text = self._read_final_output(output_path)
            attempt_trace = self._summarize_event_stream(
                stdout_path,
                stderr_path,
                temporary_secrets=provider.temporary_secrets,
            )
            return response_text, attempt_trace, provider.temporary_secrets

    async def _invoke(
        self,
        prompt: str,
        *,
        validate: ResponseValidator | None = None,
    ) -> CodexProbeResult:
        failures: list[str] = []
        attempts: list[_AttemptTrace] = []
        current_prompt = prompt
        for attempt_index in range(self.max_repair_attempts + 1):
            response_text, attempt_trace, temporary_secrets = await self._invoke_once(
                current_prompt
            )
            attempts.append(attempt_trace)
            try:
                value, canonical = self._parse_response(
                    response_text, temporary_secrets=temporary_secrets
                )
                current_trace = self._aggregate_trace([attempt_trace], failures)
                if (
                    self.tool_choice_mode == "required-first"
                    and current_trace.repo_tool_call_count < 1
                ):
                    raise self._error(
                        "SemanticValidationError",
                        "required-first attempt made no repository tool call",
                    )
                if validate is not None:
                    try:
                        semantic_error = validate(value, current_trace)
                    except Exception as exc:
                        semantic_error = (
                            f"response validator raised {type(exc).__name__}: {exc}"
                        )
                    if semantic_error is not None:
                        if (
                            not isinstance(semantic_error, str)
                            or not semantic_error.strip()
                        ):
                            semantic_error = (
                                "response validator returned an invalid "
                                "rejection reason"
                            )
                        safe_reason = self._safe_diagnostic(
                            semantic_error.encode("utf-8")
                        )
                        raise self._error("SemanticValidationError", safe_reason)
                return CodexProbeResult(
                    value=value,
                    canonical=canonical,
                    trace=self._aggregate_trace(attempts, failures),
                )
            except LLMCallError as error:
                if error.error_type not in _REPAIRABLE_ERROR_TYPES:
                    raise
                failures.append(self._failure_summary(attempt_index + 1, error))
                if attempt_index >= self.max_repair_attempts:
                    trace = self._aggregate_trace(attempts, failures)
                    trace_json = json.dumps(
                        trace.to_dict(), ensure_ascii=True, sort_keys=True
                    )
                    raise self._error(
                        error.error_type,
                        f"{self._safe_diagnostic(error.message.encode('utf-8'))}; "
                        f"structured response exhausted {len(attempts)} attempt(s); "
                        f"sanitized_trace={trace_json}",
                    ) from error
                current_prompt = self._repair_prompt(
                    prompt, response_text, error, attempt_index + 1
                )
        raise AssertionError("unreachable structured-response retry state")

    @staticmethod
    def _tracked_raw_output(result: CodexProbeResult) -> str:
        if not result.trace.failure_summaries:
            return result.canonical
        # Candidate extraction consumes ``text``. The saved raw-output field
        # can therefore retain this bounded audit envelope after a repair.
        return json.dumps(
            {
                "response": result.value,
                "simpletes_codex_audit": result.trace.to_dict(),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )

    async def capability_probe(
        self,
        prompt: str,
        *,
        validate: ResponseValidator | None = None,
    ) -> CodexProbeResult:
        """Run a schema/semantic checked probe that must inspect the repository."""

        def validate_probe(
            value: dict[str, Any], trace: CodexTraceSummary
        ) -> str | None:
            if trace.repo_tool_call_count < 1:
                return "capability probe made no repository tool call"
            return validate(value, trace) if validate is not None else None

        return await self._invoke(prompt, validate=validate_probe)

    async def preflight(
        self,
        prompt: str | None = None,
        *,
        validate: ResponseValidator | None = None,
    ) -> CodexProbeResult:
        """Exercise provider transport plus structured response validation."""
        if prompt is None:
            prompt = (
                "Return one concrete JSON object that conforms exactly to the configured "
                "response contract. This is a transport and schema validation request; "
                "do not modify the repository and do not return placeholder values."
            )
        return await self._invoke(prompt, validate=validate)

    async def generate(
        self,
        prompt: str,
        instance_id: str = "",
        track_io: bool = False,
    ) -> LLMResult:
        del instance_id
        result = await self._invoke(prompt, validate=self._response_validator)
        return LLMResult(
            text=self._marked_response(result.canonical),
            prompt=prompt if track_io else None,
            raw_output=self._tracked_raw_output(result) if track_io else None,
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


__all__ = [
    "CodexExecClient",
    "CodexProbeResult",
    "CodexTraceSummary",
    "ResponseValidator",
]
