"""HTTP helpers: expose the trace id to clients so a bug report can be traced."""

from __future__ import annotations

from typing import Any

from private_gpt.observability import setup
from private_gpt.observability.tracing import current_trace_id


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
