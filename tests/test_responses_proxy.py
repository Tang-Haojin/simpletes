from __future__ import annotations

import asyncio
import json
from urllib.parse import urlsplit

import httpx
import pytest

from simpletes.llm import responses_proxy
from simpletes.llm.responses_proxy import (
    RequiredFirstToolProxy,
    ResponsesProxyError,
    validate_upstream_base_url,
)


class _FakeResponse:
    status_code = 200
    headers = httpx.Headers(
        {
            "content-type": "text/event-stream",
            "content-encoding": "gzip",
            "openai-model": "upstream-k3",
            "x-codex-turn-state": "sticky-test-state",
            "x-models-etag": "upstream-models-etag-test",
            "x-reasoning-included": "true",
            "set-cookie": "must-not-be-forwarded=true",
        }
    )

    async def aiter_bytes(self):
        yield b'data: {"type":"response.completed"}\n\n'


class _FakeStream:
    async def __aenter__(self):
        return _FakeResponse()

    async def __aexit__(self, *_args):
        return None


class _FakeUpstreamClient:
    def __init__(self, **_kwargs) -> None:
        self.requests: list[dict[str, object]] = []
        self.closed = False

    def stream(self, method, url, *, headers, content):
        self.requests.append(
            {
                "method": method,
                "url": url,
                "headers": dict(headers),
                "body": json.loads(content),
            }
        )
        return _FakeStream()

    async def aclose(self) -> None:
        self.closed = True


async def _raw_request(
    base_url: str,
    body: dict[str, object],
    key: str,
    *,
    extra_headers: tuple[tuple[str, str], ...] = (),
) -> bytes:
    parsed = urlsplit(base_url)
    reader, writer = await asyncio.open_connection(parsed.hostname, parsed.port)
    raw_body = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = (
        b"POST /v1/responses HTTP/1.1\r\n"
        + f"Host: {parsed.hostname}:{parsed.port}\r\n".encode("ascii")
        + f"Authorization: Bearer {key}\r\n".encode("ascii")
        + b"Content-Type: application/json\r\n"
        + b"".join(
            f"{name}: {value}\r\n".encode("ascii")
            for name, value in extra_headers
        )
        + f"Content-Length: {len(raw_body)}\r\n".encode("ascii")
        + b"Connection: close\r\n\r\n"
        + raw_body
    )
    writer.write(request)
    await writer.drain()
    response = await reader.read()
    writer.close()
    await writer.wait_closed()
    return response


def test_required_first_proxy_rewrites_initial_body_retries_not_followup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_FakeUpstreamClient] = []

    def client_factory(**kwargs):
        client = _FakeUpstreamClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(responses_proxy.httpx, "AsyncClient", client_factory)

    async def exercise() -> None:
        async with RequiredFirstToolProxy(
            upstream_base_url="https://provider.example/v1",
            upstream_api_key="private-test-key",
            client_api_key="loopback-test-key",
            timeout=30,
        ) as proxy:
            assert proxy.base_url is not None
            base_url = proxy.base_url
            request = {
                "model": "k3",
                "tools": [{"type": "function", "name": "shell_command"}],
                "tool_choice": "auto",
                "parallel_tool_calls": True,
            }
            first = await _raw_request(
                base_url,
                request,
                "loopback-test-key",
                extra_headers=(
                    ("X-Codex-Turn-State", "sticky-test-state"),
                    ("X-Client-Request-Id", "client-request-test"),
                    ("Originator", "codex_cli_rs"),
                    ("OpenAI-Model", "k3"),
                    ("X-Reasoning-Included", "true"),
                    ("X-Models-Etag", "models-etag-test"),
                ),
            )
            retry = await _raw_request(base_url, request, "loopback-test-key")
            followup_request = {
                **request,
                "input": [
                    {
                        "type": "function_call_output",
                        "call_id": "call-test",
                        "output": "bounded test output",
                    }
                ],
            }
            followup = await _raw_request(
                base_url, followup_request, "loopback-test-key"
            )
            assert first.startswith(b"HTTP/1.1 200 OK")
            assert retry.startswith(b"HTTP/1.1 200 OK")
            assert followup.startswith(b"HTTP/1.1 200 OK")
            assert b"response.completed" in first
            assert b"x-codex-turn-state: sticky-test-state" in first.lower()
            assert b"openai-model: upstream-k3" in first.lower()
            assert b"x-reasoning-included: true" in first.lower()
            assert b"x-models-etag: upstream-models-etag-test" in first.lower()
            assert b"content-encoding" not in first.lower()
            assert b"set-cookie" not in first.lower()
        assert proxy.base_url is None

    asyncio.run(exercise())

    assert len(clients) == 1
    assert clients[0].closed is True
    assert [item["body"]["tool_choice"] for item in clients[0].requests] == [
        "required",
        "required",
        "auto",
    ]
    assert clients[0].requests[0]["body"]["parallel_tool_calls"] is False
    assert clients[0].requests[1]["body"]["parallel_tool_calls"] is False
    assert clients[0].requests[2]["body"]["parallel_tool_calls"] is True
    assert all(
        item["url"] == "https://provider.example/v1/responses"
        for item in clients[0].requests
    )
    assert all(
        item["headers"]["authorization"] == "Bearer private-test-key"
        for item in clients[0].requests
    )
    assert clients[0].requests[0]["headers"]["x-codex-turn-state"] == (
        "sticky-test-state"
    )
    assert clients[0].requests[0]["headers"]["x-client-request-id"] == (
        "client-request-test"
    )
    assert clients[0].requests[0]["headers"]["originator"] == "codex_cli_rs"
    assert "openai-model" not in clients[0].requests[0]["headers"]
    assert "x-reasoning-included" not in clients[0].requests[0]["headers"]
    assert "x-models-etag" not in clients[0].requests[0]["headers"]


def test_required_first_proxy_rejects_wrong_key_without_forwarding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_FakeUpstreamClient] = []

    def client_factory(**kwargs):
        client = _FakeUpstreamClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(responses_proxy.httpx, "AsyncClient", client_factory)

    async def exercise() -> bytes:
        async with RequiredFirstToolProxy(
            upstream_base_url="https://provider.example/v1",
            upstream_api_key="upstream-key",
            client_api_key="expected-key",
            timeout=None,
        ) as proxy:
            assert proxy.base_url is not None
            return await _raw_request(
                proxy.base_url,
                {"tools": [{"type": "function", "name": "repo_read"}]},
                "wrong-key",
            )

    response = asyncio.run(exercise())

    assert response.startswith(b"HTTP/1.1 401 Unauthorized")
    assert clients[0].requests == []
    assert b"expected-key" not in response


@pytest.mark.parametrize(
    "value",
    [
        "",
        "http://provider.example/v1",
        "https://user:pass@provider.example/v1",
        "https://provider.example/v1?debug=1",
        "https://provider.example:not-a-port/v1",
        "https://provider.example:70000/v1",
        "https://provider.example:0/v1",
        "relative/v1",
    ],
)
def test_validate_upstream_base_url_rejects_unsafe_values(value: str) -> None:
    with pytest.raises(ResponsesProxyError):
        validate_upstream_base_url(value)


def test_validate_upstream_base_url_normalizes_trailing_slash() -> None:
    assert (
        validate_upstream_base_url("https://provider.example/api/v1/")
        == "https://provider.example/api/v1"
    )


def test_proxy_enter_bind_failure_closes_client_and_normalizes_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clients: list[_FakeUpstreamClient] = []

    def client_factory(**kwargs):
        client = _FakeUpstreamClient(**kwargs)
        clients.append(client)
        return client

    async def fail_bind(*_args, **_kwargs):
        raise OSError("sensitive bind detail: private-test-key")

    monkeypatch.setattr(responses_proxy.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(responses_proxy.asyncio, "start_server", fail_bind)

    async def exercise() -> tuple[RequiredFirstToolProxy, str]:
        proxy = RequiredFirstToolProxy(
            upstream_base_url="https://provider.example/v1",
            upstream_api_key="private-test-key",
            client_api_key="loopback-test-key",
            timeout=30,
        )
        with pytest.raises(ResponsesProxyError) as raised:
            await proxy.__aenter__()
        return proxy, str(raised.value)

    proxy, message = asyncio.run(exercise())

    assert message == "failed to start loopback Responses proxy"
    assert "private-test-key" not in message
    assert clients[0].closed is True
    assert proxy.base_url is None


def test_proxy_enter_client_failure_is_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_client(**_kwargs):
        raise RuntimeError("sensitive client detail: private-test-key")

    monkeypatch.setattr(responses_proxy.httpx, "AsyncClient", fail_client)

    async def exercise() -> tuple[RequiredFirstToolProxy, str]:
        proxy = RequiredFirstToolProxy(
            upstream_base_url="https://provider.example/v1",
            upstream_api_key="private-test-key",
            client_api_key="loopback-test-key",
            timeout=30,
        )
        with pytest.raises(ResponsesProxyError) as raised:
            await proxy.__aenter__()
        return proxy, str(raised.value)

    proxy, message = asyncio.run(exercise())

    assert message == "failed to start loopback Responses proxy"
    assert "private-test-key" not in message
    assert proxy.base_url is None


def test_proxy_handler_catches_unexpected_error_without_leaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExplodingClient(_FakeUpstreamClient):
        def stream(self, *_args, **_kwargs):
            raise RuntimeError("sensitive upstream detail: private-test-key")

    clients: list[ExplodingClient] = []

    def client_factory(**kwargs):
        client = ExplodingClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(responses_proxy.httpx, "AsyncClient", client_factory)

    async def exercise() -> bytes:
        async with RequiredFirstToolProxy(
            upstream_base_url="https://provider.example/v1",
            upstream_api_key="private-test-key",
            client_api_key="loopback-test-key",
            timeout=30,
        ) as proxy:
            assert proxy.base_url is not None
            return await _raw_request(
                proxy.base_url,
                {
                    "model": "k3",
                    "tools": [{"type": "function", "name": "repo_read"}],
                    "tool_choice": "auto",
                },
                "loopback-test-key",
            )

    response = asyncio.run(exercise())

    assert response.startswith(b"HTTP/1.1 502 Bad Gateway")
    assert b"sensitive upstream detail" not in response
    assert b"private-test-key" not in response
    assert b"loopback-test-key" not in response
    assert clients[0].closed is True


def test_proxy_task_done_retrieves_exception() -> None:
    class DoneTask:
        exception_retrieved = False

        def cancelled(self) -> bool:
            return False

        def exception(self) -> RuntimeError:
            self.exception_retrieved = True
            return RuntimeError("sensitive task detail")

    proxy = RequiredFirstToolProxy(
        upstream_base_url="https://provider.example/v1",
        upstream_api_key="private-test-key",
        client_api_key="loopback-test-key",
        timeout=30,
    )
    task = DoneTask()
    proxy._tasks.add(task)  # type: ignore[arg-type]

    proxy._task_done(task)  # type: ignore[arg-type]

    assert task.exception_retrieved is True
    assert task not in proxy._tasks
