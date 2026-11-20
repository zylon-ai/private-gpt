import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from aiohttp import web

from private_gpt.components.web.web_search.providers import parallel
from private_gpt.settings.settings import (
    NetworkSettings,
    ParallelSearchSettings,
    Settings,
)


def _provider(api_key: str | None = None, timeout: int = 5, offline: bool = False):
    settings = MagicMock(spec=Settings)
    settings.parallel = ParallelSearchSettings(api_key=api_key, timeout=timeout)
    settings.server = MagicMock()
    settings.server.network = NetworkSettings(offline_mode=offline)
    return parallel.ParallelSearchProvider(settings)


async def test_parallel_validation_respects_offline_mode():
    with pytest.raises(ValueError, match="unavailable in offline mode"):
        await _provider(offline=True).validate()


@asynccontextmanager
async def _serve(handler: Any) -> AsyncIterator[str]:
    app = web.Application()
    app.router.add_route("*", "/mcp", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{runner.addresses[0][1]}/mcp"
    finally:
        await runner.cleanup()


async def _rpc_response(call: dict, result: dict):
    return web.json_response(
        {"jsonrpc": "2.0", "id": call["id"], "result": result},
        headers={"Mcp-Session-Id": "test-session"}
        if call["method"] == "initialize"
        else {},
    )


async def _handle_rpc(call: dict, search_hit: dict):
    method = call["method"]
    if method == "initialize":
        return await _rpc_response(
            call,
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "test", "version": "1"},
            },
        )
    if method == "notifications/initialized":
        return web.Response(status=202)
    if method == "tools/list":
        return await _rpc_response(
            call,
            {
                "tools": [{"name": "web_search", "inputSchema": {"type": "object"}}],
            },
        )
    return await _rpc_response(
        call,
        {
            "content": [],
            "structuredContent": {"results": [search_hit, search_hit]},
        },
    )


@pytest.mark.parametrize("api_key", [None, "synthetic-key"])
async def test_parallel_search_uses_sdk_and_maps_excerpts(monkeypatch, api_key):
    requests = []
    hit = {
        "url": "https://example.com/voyager",
        "title": None,
        "publish_date": "2026-01-01",
        "excerpts": ["First excerpt.", "Second excerpt."],
    }

    async def handler(request):
        call = await request.json() if request.method == "POST" else None
        requests.append((request.method, call, request.headers))
        if request.method == "GET":
            return web.Response(status=405)
        if request.method == "DELETE":
            return web.Response(status=204)
        return await _handle_rpc(call, hit)

    async with _serve(handler) as url:
        monkeypatch.setattr(parallel, "_PARALLEL_MCP_URL", url)
        results = await _provider(api_key).make_query("  Voyager mission  ", 1)

    search = next(
        call for _, call, _ in requests if call and call["method"] == "tools/call"
    )
    assert search["params"]["name"] == "web_search"
    assert search["params"]["arguments"] == {
        "objective": "Voyager mission",
        "search_queries": ["Voyager mission"],
    }
    assert {call["method"] for _, call, _ in requests if call} == {
        "initialize",
        "notifications/initialized",
        "tools/list",
        "tools/call",
    }
    assert requests[-1][0] == "DELETE"
    assert all(headers["User-Agent"] == "PrivateGPT" for _, _, headers in requests)
    expected_auth = f"Bearer {api_key}" if api_key else None
    assert all(
        headers.get("Authorization") == expected_auth for _, _, headers in requests
    )
    assert [(r.url, r.title, r.description, r.age) for r in results] == [
        (hit["url"], hit["url"], "First excerpt.\n\nSecond excerpt.", "2026-01-01"),
    ]


@pytest.mark.parametrize("redirect_on", ["tools/call", "DELETE"])
async def test_parallel_keeps_query_and_session_on_original_origin(
    monkeypatch, redirect_on
):
    forwarded = []

    async def destination(request):
        forwarded.append(request.method)
        return web.Response()

    async with _serve(destination) as redirect:

        async def handler(request):
            if request.method == "GET":
                return web.Response(status=405)
            if request.method == "DELETE":
                if redirect_on == "DELETE":
                    raise web.HTTPTemporaryRedirect(location=redirect)
                return web.Response(status=204)
            call = await request.json()
            if call["method"] == redirect_on:
                raise web.HTTPTemporaryRedirect(location=redirect)
            return await _handle_rpc(
                call,
                {
                    "url": "https://example.com/",
                    "excerpts": ["An excerpt"],
                },
            )

        async with _serve(handler) as url:
            monkeypatch.setattr(parallel, "_PARALLEL_MCP_URL", url)
            if redirect_on == "tools/call":
                with pytest.raises(RuntimeError, match="Parallel web search failed"):
                    await _provider("synthetic-key").make_query("Private query", 1)
            else:
                assert await _provider("synthetic-key").make_query("Private query", 1)
    assert forwarded == []


@pytest.mark.parametrize("cancel", [False, True])
async def test_parallel_transports_close_on_timeout_or_cancellation(
    monkeypatch, cancel
):
    started = asyncio.Event()
    release = asyncio.Event()
    closed = asyncio.Event()

    async def handler(request):
        if request.method == "GET":
            return web.Response(status=405)
        if request.method == "DELETE":
            closed.set()
            return web.Response(status=204)
        call = await request.json()
        if call["method"] == "tools/call":
            started.set()
            await release.wait()
        return await _handle_rpc(call, {})

    async with _serve(handler) as url:
        monkeypatch.setattr(parallel, "_PARALLEL_MCP_URL", url)
        task = asyncio.create_task(_provider(timeout=1).make_query("Search query", 1))
        try:
            await asyncio.wait_for(started.wait(), 3)
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(TimeoutError, match="Parallel web search timed out"):
                    await task
            assert closed.is_set()
        finally:
            release.set()


@pytest.mark.parametrize("tool_error", [False, True])
async def test_parallel_closes_held_open_sse_reply(monkeypatch, tool_error):
    release = asyncio.Event()

    async def handler(request):
        if request.method == "GET":
            return web.Response(status=405)
        if request.method == "DELETE":
            return web.Response(status=204)
        call = await request.json()
        if call["method"] != "tools/call":
            return await _handle_rpc(call, {})

        data = {"jsonrpc": "2.0", "id": call["id"]}
        if tool_error:
            data["error"] = {"code": -32000, "message": "Search limit reached"}
        else:
            data["result"] = {"content": [], "structuredContent": {"results": []}}
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(f"data: {json.dumps(data)}\r\n\r\n".encode())
        await release.wait()
        return response

    async with _serve(handler) as url:
        monkeypatch.setattr(parallel, "_PARALLEL_MCP_URL", url)
        try:
            if tool_error:
                with pytest.raises(RuntimeError, match="Search limit reached"):
                    await asyncio.wait_for(_provider().make_query("Search query", 1), 3)
            else:
                assert (
                    await asyncio.wait_for(_provider().make_query("Search query", 1), 3)
                    == []
                )
        finally:
            release.set()


@pytest.mark.parametrize("stream_body", [False, True])
async def test_parallel_session_teardown_has_total_deadline(monkeypatch, stream_body):
    release = asyncio.Event()

    async def handler(request):
        if request.method == "GET":
            return web.Response(status=405)
        if request.method == "DELETE":
            if not stream_body:
                await release.wait()
                return web.Response(status=204)
            response = web.StreamResponse(status=200)
            await response.prepare(request)
            try:
                for _ in range(12):
                    await response.write(b" ")
                    await asyncio.sleep(0.25)
            except ConnectionResetError:
                pass
            return response
        call = await request.json()
        return await _handle_rpc(
            call,
            {
                "url": "https://example.com/",
                "excerpts": ["An excerpt"],
            },
        )

    async with _serve(handler) as url:
        monkeypatch.setattr(parallel, "_PARALLEL_MCP_URL", url)
        try:
            results = await asyncio.wait_for(
                _provider().make_query("Search query", 1), 2
            )
            assert results[0].description == "An excerpt"
        finally:
            release.set()
