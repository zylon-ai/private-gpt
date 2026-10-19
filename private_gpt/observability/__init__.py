"""Vendor-neutral tracing for PrivateGPT.

Application code only talks to the plain OpenTelemetry API through the helpers
in :mod:`private_gpt.observability.tracing`. Where spans end up (Phoenix,
Langfuse, LangSmith, Opik, Grafana Tempo, Datadog, ...) is decided purely by
``observability.exporters`` in settings, see
:mod:`private_gpt.observability.setup`.

When tracing is disabled (or the ``observability-otel`` extra is missing) every
helper degrades to a no-op, so instrumented code never needs to guard itself.
"""

from private_gpt.observability.tracing import (
    SpanKind,
    current_trace_id,
    inject_trace_context,
    record_exception,
    set_span_attributes,
    start_span,
    trace_context_from,
    using_trace_attributes,
)

__all__ = [
    "SpanKind",
    "current_trace_id",
    "inject_trace_context",
    "record_exception",
    "set_span_attributes",
    "start_span",
    "trace_context_from",
    "using_trace_attributes",
]
