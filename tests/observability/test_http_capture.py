"""HTTP server span: request/response bodies, SSE, status and kind."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

from private_gpt.observability import setup
from private_gpt.observability.http import HttpBodyCaptureMiddleware
from private_gpt.settings.settings import ObservabilitySettings

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from opentelemetry.sdk.trace import ReadableSpan

pytestmark = pytest.mark.anyio


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    setup._configured = True
    setup._settings = ObservabilitySettings(
        mode="otel",
        capture_http_paths=["/v1/*"],
        max_http_body_bytes=200,
        excluded_urls=["/health"],
    )
    try:
        yield memory
    finally:
        setup._reset_for_tests()
        trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]


def _sse(*events: tuple[str, dict]) -> list[bytes]:
    return [
        f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in events
    ]


def _app(provider: TracerProvider, sent: list[bytes]) -> FastAPI:
    app = FastAPI()

    @app.post("/v1/messages")
    async def messages(body: dict) -> JSONResponse:
        if body.get("bad"):
            return JSONResponse({"type": "error", "message": "invalid"}, 400)
        return JSONResponse({"content": [{"type": "text", "text": "hi"}]})

    @app.post("/v1/stream")
    async def stream(body: dict) -> StreamingResponse:
        events = [
            ("message_start", {"type": "message_start"}),
            ("content_block_delta", {"type": "content_block_delta", "text": "h"}),
        ]
        if body.get("fail"):
            events.append(("error", {"type": "error", "error": {"message": "boom"}}))
        else:
            events.append(("message_stop", {"type": "message_stop"}))
        chunks = _sse(*events)

        async def gen() -> AsyncIterator[bytes]:
            for chunk in chunks:
                sent.append(chunk)
                yield chunk

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.post("/v1/big")
    async def big() -> JSONResponse:
        return JSONResponse({"text": "x" * 1000})

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    app.add_middleware(HttpBodyCaptureMiddleware)
    FastAPIInstrumentor.instrument_app(
        app, tracer_provider=provider, exclude_spans=["send", "receive"]
    )
    return app


async def _call(
    exporter: InMemorySpanExporter, method: str, path: str, **kwargs: object
) -> tuple[httpx.Response, ReadableSpan | None, list[bytes]]:
    sent: list[bytes] = []
    app = _app(trace.get_tracer_provider(), sent)  # type: ignore[arg-type]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        response = await client.request(method, path, **kwargs)
    spans = [s for s in exporter.get_finished_spans() if s.name.endswith(path)]
    return response, (spans[0] if spans else None), sent


async def test_json_request_and_response_are_recorded_ok(
    exporter: InMemorySpanExporter,
) -> None:
    response, span, _ = await _call(
        exporter, "POST", "/v1/messages", json={"messages": [{"content": "hello"}]}
    )

    assert response.status_code == 200
    assert span is not None
    assert span.attributes["openinference.span.kind"] == "CHAIN"
    assert json.loads(span.attributes["input.value"]) == {
        "messages": [{"content": "hello"}]
    }
    assert json.loads(span.attributes["output.value"])["content"][0]["text"] == "hi"
    assert span.status.status_code == StatusCode.OK


async def test_client_error_marks_span_error(exporter: InMemorySpanExporter) -> None:
    _, span, _ = await _call(exporter, "POST", "/v1/messages", json={"bad": True})

    assert span is not None
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == "HTTP 400"
    assert '"invalid"' in span.attributes["output.value"]


async def test_sse_stream_is_forwarded_unchanged_and_recorded(
    exporter: InMemorySpanExporter,
) -> None:
    response, span, sent = await _call(exporter, "POST", "/v1/stream", json={})

    # The client received exactly the bytes the app produced.
    assert response.content == b"".join(sent)
    assert span is not None
    assert span.attributes["pgpt.http.streaming"] is True
    assert "event: message_stop" in span.attributes["output.value"]
    assert span.status.status_code == StatusCode.OK


async def test_error_event_inside_200_stream_marks_span_error(
    exporter: InMemorySpanExporter,
) -> None:
    response, span, _ = await _call(exporter, "POST", "/v1/stream", json={"fail": True})

    assert response.status_code == 200
    assert span is not None
    assert span.status.status_code == StatusCode.ERROR
    assert "boom" in span.status.description


async def test_large_body_is_capped_but_size_counted(
    exporter: InMemorySpanExporter,
) -> None:
    response, span, _ = await _call(exporter, "POST", "/v1/big")

    assert len(response.content) > 1000
    assert span is not None
    assert span.attributes["http.response.body.truncated"] is True
    assert span.attributes["http.response.body.size"] == len(response.content)
    assert len(span.attributes["output.value"]) <= 200
    assert span.status.status_code == StatusCode.OK


async def test_excluded_paths_are_not_captured(
    exporter: InMemorySpanExporter,
) -> None:
    _, span, _ = await _call(exporter, "GET", "/health")

    assert span is None or "output.value" not in (span.attributes or {})


async def test_bodies_are_not_captured_when_content_capture_is_off(
    exporter: InMemorySpanExporter,
) -> None:
    setup._capture_content = False
    _, span, _ = await _call(exporter, "POST", "/v1/messages", json={"q": "secret"})

    assert span is not None
    assert "input.value" not in span.attributes
    assert span.attributes["http.request.body.size"] == 0
    # Status and kind are still set.
    assert span.status.status_code == StatusCode.OK


def test_patterns_match_routes_behind_a_root_path() -> None:
    from private_gpt.observability.http import _route_path

    scope = {"path": "/gpt/v1/messages/count_tokens", "root_path": "/gpt"}
    assert _route_path(scope) == "/v1/messages/count_tokens"
    assert _route_path({"path": "/v1/messages", "root_path": ""}) == "/v1/messages"
    assert _route_path({"path": "/gptx/v1", "root_path": "/gpt"}) == "/gptx/v1"


def test_inline_files_are_replaced_by_their_size() -> None:
    from private_gpt.observability.http import _shape

    payload = "QUJD" * 300  # 1200 base64 chars -> 900 bytes
    body = json.dumps({"input": {"type": "file", "value": payload}, "n": 1})
    assert _shape(body, "application/json") == {
        "input": {"type": "file", "value": "<base64, 900 bytes>"},
        "n": 1,
    }
    # Cut by the capture cap: no closing quote, still elided.
    assert _shape('{"value": "' + payload, "application/json") == (
        '{"value": "<base64, 900+ bytes>"'
    )
