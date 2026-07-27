"""Small loopback proxy for narrowly-scoped Responses API compatibility.

Some OpenAI-compatible providers only enter a useful agent tool loop when the
first Responses request uses ``tool_choice=required``.  Codex intentionally
sends ``tool_choice=auto`` and does not expose that request field as a public
configuration option.  This proxy remembers only a digest of the first
accepted request body: byte-identical transport retries remain required, while
a different tool-output follow-up is forwarded unchanged.

The proxy binds an ephemeral IPv4 loopback port, authenticates inbound requests
with an independent per-attempt loopback credential, accepts only
``POST /v1/responses``, and never records request or response bodies.  It is not
a general HTTP proxy.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
from contextlib import AbstractAsyncContextManager
from http import HTTPStatus
from typing import Any
from urllib.parse import urlsplit

import httpx


_MAX_HEADER_BYTES = 64 * 1024
_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024
_LOCAL_PATH = "/v1/responses"
_REQUEST_HEADER_ALLOWLIST = frozenset(
    {
        "accept",
        "content-type",
        "originator",
        "user-agent",
        "openai-beta",
        "session-id",
        "thread-id",
        "x-client-request-id",
    }
)
_REQUEST_HEADER_PREFIX_ALLOWLIST = (
    "x-codex-",
    "x-openai-",
    "x-responsesapi-",
)
_RESPONSE_HEADER_ALLOWLIST = frozenset(
    {
        "content-type",
        "openai-model",
        "retry-after",
        "x-models-etag",
        "x-reasoning-included",
        "x-request-id",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
    }
)
_RESPONSE_HEADER_PREFIX_ALLOWLIST = ("x-codex-", "x-ratelimit-")


def _request_header_allowed(name: str) -> bool:
    return name in _REQUEST_HEADER_ALLOWLIST or name.startswith(
        _REQUEST_HEADER_PREFIX_ALLOWLIST
    )


def _response_header_allowed(name: str) -> bool:
    return name in _RESPONSE_HEADER_ALLOWLIST or name.startswith(
        _RESPONSE_HEADER_PREFIX_ALLOWLIST
    )


class ResponsesProxyError(RuntimeError):
    """Raised for invalid setup or a bounded loopback proxy failure."""


def validate_upstream_base_url(value: str) -> str:
    """Return a normalized HTTPS API base URL safe for credential forwarding."""
    if not isinstance(value, str) or not value.strip():
        raise ResponsesProxyError("provider base_url must be a non-empty string")
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        raise ResponsesProxyError(
            "provider base_url must be an absolute HTTPS URL"
        ) from None
    if parsed.scheme != "https" or not parsed.hostname:
        raise ResponsesProxyError("provider base_url must be an absolute HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ResponsesProxyError("provider base_url must not contain user information")
    if parsed.query or parsed.fragment:
        raise ResponsesProxyError("provider base_url must not contain a query or fragment")
    try:
        port = parsed.port
    except ValueError:
        raise ResponsesProxyError("provider base_url has an invalid port") from None
    if port is not None and not 1 <= port <= 65535:
        raise ResponsesProxyError("provider base_url has an invalid port")
    return value.strip().rstrip("/")


class RequiredFirstToolProxy(AbstractAsyncContextManager["RequiredFirstToolProxy"]):
    """Forward one Codex attempt while requiring a tool on its first request."""

    def __init__(
        self,
        *,
        upstream_base_url: str,
        upstream_api_key: str,
        client_api_key: str,
        timeout: float | None,
    ) -> None:
        self._upstream_base_url = validate_upstream_base_url(upstream_base_url)
        if not isinstance(upstream_api_key, str) or not upstream_api_key:
            raise ResponsesProxyError("upstream API key must be a non-empty string")
        if not isinstance(client_api_key, str) or not client_api_key:
            raise ResponsesProxyError("client API key must be a non-empty string")
        if secrets.compare_digest(upstream_api_key, client_api_key):
            raise ResponsesProxyError("client API key must differ from upstream key")
        self._upstream_api_key = upstream_api_key
        self.client_api_key = client_api_key
        self._timeout = timeout
        self._server: asyncio.AbstractServer | None = None
        self._client: httpx.AsyncClient | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        self._first_request_digest: bytes | None = None
        self._first_request_lock = asyncio.Lock()
        self.base_url: str | None = None

    async def __aenter__(self) -> "RequiredFirstToolProxy":
        try:
            timeout = httpx.Timeout(self._timeout) if self._timeout is not None else None
            self._client = httpx.AsyncClient(
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
            )
            self._server = await asyncio.start_server(
                self._accept,
                host="127.0.0.1",
                port=0,
                limit=_MAX_HEADER_BYTES,
            )
            sockets = self._server.sockets or []
            if len(sockets) != 1:
                raise ResponsesProxyError(
                    "loopback proxy did not bind exactly one socket"
                )
            sockname = sockets[0].getsockname()
            if (
                not isinstance(sockname, tuple)
                or len(sockname) < 2
                or isinstance(sockname[1], bool)
                or not isinstance(sockname[1], int)
                or not 1 <= sockname[1] <= 65535
            ):
                raise ResponsesProxyError("loopback proxy bound an invalid port")
            self.base_url = f"http://127.0.0.1:{sockname[1]}/v1"
            return self
        except asyncio.CancelledError:
            await self._cleanup()
            raise
        except ResponsesProxyError:
            await self._cleanup()
            raise
        except Exception:
            await self._cleanup()
            raise ResponsesProxyError("failed to start loopback Responses proxy") from None

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self._cleanup()

    async def _cleanup(self) -> None:
        if self._server is not None:
            server = self._server
            self._server = None
            try:
                server.close()
                await server.wait_closed()
            except Exception:
                pass
        active = tuple(self._tasks)
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        self._tasks.clear()
        if self._client is not None:
            client = self._client
            self._client = None
            try:
                await client.aclose()
            except Exception:
                pass
        self.base_url = None

    def _accept(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        task = asyncio.create_task(self._handle_connection(reader, writer))
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        try:
            task.exception()
        except BaseException:
            # Retrieving the exception is sufficient; connection handlers use
            # bounded client responses instead of process-level diagnostics.
            pass

    @staticmethod
    async def _write_error(
        writer: asyncio.StreamWriter, status: HTTPStatus, message: str
    ) -> None:
        body = json.dumps(
            {"error": {"message": message, "type": "simpletes_proxy_error"}},
            separators=(",", ":"),
        ).encode("utf-8")
        writer.write(
            f"HTTP/1.1 {status.value} {status.phrase}\r\n".encode("ascii")
            + b"Content-Type: application/json\r\n"
            + f"Content-Length: {len(body)}\r\n".encode("ascii")
            + b"Connection: close\r\n\r\n"
            + body
        )
        await writer.drain()

    @staticmethod
    def _parse_headers(raw: bytes) -> tuple[str, str, dict[str, str]]:
        try:
            text = raw.decode("iso-8859-1")
            lines = text.split("\r\n")
            method, path, version = lines[0].split(" ", 2)
        except (UnicodeError, ValueError, IndexError) as error:
            raise ResponsesProxyError("malformed loopback HTTP request") from error
        if version != "HTTP/1.1":
            raise ResponsesProxyError("loopback proxy requires HTTP/1.1")
        headers: dict[str, str] = {}
        for line in lines[1:]:
            if not line:
                continue
            if line[0] in " \t" or ":" not in line:
                raise ResponsesProxyError("malformed loopback HTTP header")
            name, value = line.split(":", 1)
            normalized = name.strip().lower()
            if not normalized or normalized in headers:
                raise ResponsesProxyError("duplicate or empty loopback HTTP header")
            headers[normalized] = value.strip()
        return method, path, headers

    async def _require_first_tool(
        self, body: dict[str, Any], request_digest: bytes
    ) -> bool:
        async with self._first_request_lock:
            if self._first_request_digest is None:
                tools = body.get("tools")
                if not isinstance(tools, list) or not tools:
                    raise ResponsesProxyError(
                        "first Responses request contained no tools"
                    )
                self._first_request_digest = request_digest
            if not secrets.compare_digest(
                request_digest, self._first_request_digest
            ):
                return False
            tools = body.get("tools")
            if not isinstance(tools, list) or not tools:
                raise ResponsesProxyError("first Responses request contained no tools")
            body["tool_choice"] = "required"
            body["parallel_tool_calls"] = False
            return True

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        response_started = False
        try:
            try:
                raw_headers = await reader.readuntil(b"\r\n\r\n")
            except (asyncio.IncompleteReadError, asyncio.LimitOverrunError):
                await self._write_error(
                    writer, HTTPStatus.BAD_REQUEST, "invalid loopback request headers"
                )
                return
            if len(raw_headers) > _MAX_HEADER_BYTES:
                await self._write_error(
                    writer,
                    HTTPStatus.REQUEST_HEADER_FIELDS_TOO_LARGE,
                    "loopback request headers were too large",
                )
                return
            try:
                method, path, headers = self._parse_headers(raw_headers[:-4])
                if method != "POST" or path != _LOCAL_PATH:
                    await self._write_error(
                        writer, HTTPStatus.FORBIDDEN, "loopback route is not allowed"
                    )
                    return
                if "transfer-encoding" in headers:
                    raise ResponsesProxyError(
                        "chunked loopback requests are not supported"
                    )
                raw_length = headers.get("content-length")
                if raw_length is None:
                    raise ResponsesProxyError("loopback request has no content length")
                length = int(raw_length)
                if length < 1 or length > _MAX_REQUEST_BYTES:
                    raise ResponsesProxyError("loopback request body size is invalid")
                expected_auth = f"Bearer {self.client_api_key}"
                if not secrets.compare_digest(
                    headers.get("authorization", ""), expected_auth
                ):
                    await self._write_error(
                        writer, HTTPStatus.UNAUTHORIZED, "loopback authorization failed"
                    )
                    return
                raw_body = await reader.readexactly(length)
                body = json.loads(raw_body)
                if not isinstance(body, dict):
                    raise ResponsesProxyError("Responses request body must be an object")
                request_digest = hashlib.sha256(raw_body).digest()
                await self._require_first_tool(body, request_digest)
                rewritten = json.dumps(
                    body, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
            except (
                ValueError,
                json.JSONDecodeError,
                asyncio.IncompleteReadError,
                ResponsesProxyError,
            ):
                await self._write_error(
                    writer, HTTPStatus.BAD_REQUEST, "invalid loopback Responses request"
                )
                return

            client = self._client
            if client is None:
                await self._write_error(
                    writer, HTTPStatus.SERVICE_UNAVAILABLE, "loopback proxy is closing"
                )
                return
            outbound_headers = {
                name: value
                for name, value in headers.items()
                if _request_header_allowed(name)
            }
            outbound_headers["authorization"] = f"Bearer {self._upstream_api_key}"
            outbound_headers["content-type"] = "application/json"
            upstream_url = f"{self._upstream_base_url}/responses"
            async with client.stream(
                "POST",
                upstream_url,
                headers=outbound_headers,
                content=rewritten,
            ) as response:
                try:
                    reason = HTTPStatus(response.status_code).phrase
                except ValueError:
                    reason = "Upstream Response"
                response_started = True
                writer.write(
                    f"HTTP/1.1 {response.status_code} {reason}\r\n".encode("ascii")
                )
                for name, value in response.headers.multi_items():
                    if _response_header_allowed(name.lower()):
                        writer.write(f"{name}: {value}\r\n".encode("iso-8859-1"))
                writer.write(b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n")
                await writer.drain()
                response_bytes = 0
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    response_bytes += len(chunk)
                    if response_bytes > _MAX_RESPONSE_BYTES:
                        raise ResponsesProxyError(
                            "upstream Responses body exceeded the size limit"
                        )
                    writer.write(f"{len(chunk):X}\r\n".encode("ascii"))
                    writer.write(chunk)
                    writer.write(b"\r\n")
                    await writer.drain()
                writer.write(b"0\r\n\r\n")
                await writer.drain()
        except asyncio.CancelledError:
            raise
        except Exception:
            # Provider/client exceptions may include headers, URLs, or response
            # fragments.  Never reflect their text to Codex or the process log.
            if not response_started:
                try:
                    if not writer.is_closing():
                        await self._write_error(
                            writer,
                            HTTPStatus.BAD_GATEWAY,
                            "upstream Responses request failed",
                        )
                except Exception:
                    pass
        finally:
            try:
                writer.close()
            except Exception:
                pass
            try:
                await writer.wait_closed()
            except Exception:
                pass


__all__ = [
    "RequiredFirstToolProxy",
    "ResponsesProxyError",
    "validate_upstream_base_url",
]
