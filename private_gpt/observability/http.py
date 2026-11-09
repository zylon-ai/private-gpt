"""HTTP helpers: trace ids for clients and request/response capture on spans.

``HttpBodyCaptureMiddleware`` is a pure ASGI middleware that sits inside the
FastAPI server span. It copies the request body and the response body (plain
JSON or SSE) into the span as they stream, without buffering or holding back
what the client receives: each chunk is passed through as is and a capped copy
is kept on the side. With the last chunk it sets an OpenInference span kind,
``input``/``output`` and an OK/ERROR status (HTTP 4xx/5xx, or an SSE ``error``
event inside a 200 stream).
"""

from __future__ import annotations

import fnmatch
import json
import re
from typing import TYPE_CHECKING, Any

from private_gpt.observability import setup
from private_gpt.observability.tracing import (
    SpanKind,
    current_trace_id,
    set_span_attributes,
    set_span_status,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, MutableMapping

    Scope = MutableMapping[str, Any]
    Message = MutableMapping[str, Any]
    Receive = Callable[[], Awaitable[Message]]
    Send = Callable[[Message], Awaitable[None]]
    ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def add_trace_response_headers(response: Any) -> None:
    """Add ``X-Trace-Id`` and W3C ``traceresponse`` to ``response``.

    Must run inside the server span (i.e. from an HTTP middleware). For SSE
    the headers are sent before the first event, so streaming clients get the
    id too.
    """
    obs = setup.observability_settings()
    if obs is None or not obs.trace_response_header:
        return
    trace_id = current_trace_id()
    if trace_id is None or not hasattr(response, "headers"):
        return
    from opentelemetry import trace

    ctx = trace.get_current_span().get_span_context()
    response.headers["x-trace-id"] = trace_id
    response.headers["traceresponse"] = (
        f"00-{trace_id}-{ctx.span_id:016x}-{int(ctx.trace_flags):02x}"
    )


class _CappedBuffer:
    """Keeps the first ``limit`` bytes; counts the rest."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._chunks: list[bytes] = []
        self._kept = 0
        self.total = 0

    def add(self, chunk: bytes) -> None:
        self.total += len(chunk)
        room = self._limit - self._kept
        if room > 0 and chunk:
            piece = chunk[:room]
            self._chunks.append(piece)
            self._kept += len(piece)

    @property
    def truncated(self) -> bool:
        return self.total > self._kept

    def text(self) -> str:
        return b"".join(self._chunks).decode("utf-8", errors="replace")


class _SseErrorSniffer:
    """Spots ``event: error`` / ``"type": "error"`` in an SSE stream.

    Works on raw chunks with a small carry-over, so it sees every event even
    when the capture buffer is full and costs nothing per chunk beyond a
    substring search.
    """

    _MARKERS = (b"event: error", b'"type":"error"', b'"type": "error"')

    def __init__(self) -> None:
        self._tail = b""
        self.error: str | None = None

    def feed(self, chunk: bytes) -> None:
        if self.error is not None or not chunk:
            return
        window = self._tail + chunk
        for marker in self._MARKERS:
            index = window.find(marker)
            if index != -1:
                end = window.find(b"\n\n", index)
                self.error = window[index : end if end != -1 else index + 2000].decode(
                    "utf-8", errors="replace"
                )
                return
        self._tail = window[-64:]


def _matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _route_path(scope: Scope) -> str:
    """Request path without the mount prefix (``/gpt/v1/x`` -> ``/v1/x``).

    Patterns are written against the API routes; behind a ``root_path``
    Starlette keeps the prefix in ``scope["path"]``.
    """
    path = scope.get("path", "")
    root = (scope.get("root_path") or "").rstrip("/")
    if root and path.startswith(root + "/"):
        return path[len(root) :]
    return path


# A JSON string of base64 (file uploads, images); also matches one cut by the
# capture cap, which has no closing quote.
_BASE64_STRING = re.compile(r'"([A-Za-z0-9+/]{512,}={0,2})("|$)')


def _elide_base64(text: str) -> str:
    def marker(match: re.Match[str]) -> str:
        size = len(match.group(1)) * 3 // 4
        cut = "" if match.group(2) else "+"
        return f'"<base64, {size}{cut} bytes>"'

    return _BASE64_STRING.sub(marker, text)


def _shape(text: str, content_type: str) -> Any:
    """JSON bodies are recorded as JSON; everything else as text.

    Inline files (base64) are replaced by a size marker: they are what makes
    bodies big and they are unreadable in a trace.
    """
    text = _elide_base64(text)
    if "json" in content_type:
        try:
            return json.loads(text)
        except ValueError:
            return text
    return text


class HttpBodyCaptureMiddleware:
    """Record request/response bodies and status on the HTTP server span."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        obs = setup.observability_settings()
        if (
            scope["type"] != "http"
            or obs is None
            or not setup.is_configured()
            or _matches(_route_path(scope), obs.excluded_urls)
        ):
            await self.app(scope, receive, send)
            return

        from opentelemetry import trace

        span = trace.get_current_span()
        if not span.is_recording():
            await self.app(scope, receive, send)
            return

        capture = (
            setup.capture_content()
            and obs.capture_http_bodies
            and _matches(_route_path(scope), obs.capture_http_paths)
        )
        limit = obs.max_http_body_bytes
        request_body = _CappedBuffer(limit)
        response_body = _CappedBuffer(limit)
        sniffer = _SseErrorSniffer()
        state: dict[str, Any] = {"status": None, "content_type": ""}

        async def receive_wrapper() -> Message:
            message = await receive()
            if capture and message["type"] == "http.request":
                request_body.add(message.get("body", b""))
            return message

        def finish() -> None:
            if not state.get("finished"):
                state["finished"] = True
                self._finish(span, state, request_body, response_body, sniffer)

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                for key, value in message.get("headers", []):
                    if key.lower() == b"content-type":
                        state["content_type"] = value.decode("latin-1").lower()
            elif message["type"] == "http.response.body":
                chunk = message.get("body", b"")
                if "text/event-stream" in state["content_type"]:
                    sniffer.feed(chunk)
                if capture:
                    response_body.add(chunk)
                if not message.get("more_body", False):
                    # The server span ends as soon as the last chunk is sent,
                    # so the result is recorded just before forwarding it.
                    finish()
            await send(message)

        try:
            await self.app(scope, receive_wrapper, send_wrapper)
        finally:
            finish()

    @staticmethod
    def _finish(
        span: Any,
        state: dict[str, Any],
        request_body: _CappedBuffer,
        response_body: _CappedBuffer,
        sniffer: _SseErrorSniffer,
    ) -> None:
        content_type = state["content_type"]
        streaming = "text/event-stream" in content_type
        attributes: dict[str, Any] = {
            "openinference.span.kind": str(SpanKind.CHAIN),
            "http.request.body.size": request_body.total,
            "http.response.body.size": response_body.total,
            "pgpt.http.streaming": streaming,
        }
        if request_body.total:
            attributes["input"] = _shape(request_body.text(), "json")
            attributes["http.request.body.truncated"] = request_body.truncated
        if response_body.total:
            attributes["output"] = (
                response_body.text()
                if streaming
                else _shape(response_body.text(), content_type)
            )
            attributes["http.response.body.truncated"] = response_body.truncated
        set_span_attributes(span, **attributes)

        status = state["status"]
        if status is None:
            # Disconnected or failed before responding; an exception (if any)
            # is recorded by the FastAPI instrumentation.
            return
        if status >= 400:
            set_span_status(span, error=f"HTTP {status}")
        elif sniffer.error is not None:
            set_span_status(span, error=f"Stream error: {sniffer.error[:400]}")
        else:
            set_span_status(span)
