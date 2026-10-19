"""Span processors applied once, before spans fan out to every exporter."""

from __future__ import annotations

import fnmatch
import json
import logging
import re
import threading
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import ReadableSpan, Span

logger = logging.getLogger(__name__)

_TOOL_SCHEMA_KEY = re.compile(r"^llm\.tools\.(\d+)\.tool\.json_schema$")
_BROKEN_REPR = re.compile(r'^"?<[\w.]+ object at 0x[0-9a-f]+>"?$')

# Tool definitions registered by the span that issues an LLM call (the agent
# iteration). The OpenInference llama-index handler serializes FunctionTool
# objects with ``str()`` which yields ``<FunctionTool object at 0x...>``; we
# replace those with the real JSON schema on the child LLM span.
_tools_by_parent: dict[int, list[str]] = {}
_tools_lock = threading.Lock()


def register_llm_tools(span: Any, tools: list[dict[str, Any]]) -> None:
    """Expose ``tools`` (OpenAI function format) to LLM spans under ``span``."""
    ctx = span.get_span_context() if hasattr(span, "get_span_context") else None
    if ctx is None or not ctx.is_valid:
        return
    serialized = [json.dumps(t, default=str, ensure_ascii=False) for t in tools]
    with _tools_lock:
        _tools_by_parent[ctx.span_id] = serialized


def unregister_llm_tools(span: Any) -> None:
    ctx = span.get_span_context() if hasattr(span, "get_span_context") else None
    if ctx is None:
        return
    with _tools_lock:
        _tools_by_parent.pop(ctx.span_id, None)


def _with_attributes(span: ReadableSpan, attributes: dict[str, Any]) -> ReadableSpan:
    from opentelemetry.sdk.trace import ReadableSpan

    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=attributes,
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class FanOutSpanProcessor:
    """Forward to several processors (one per exporter)."""

    def __init__(self, processors: list[Any]) -> None:
        self._processors = processors

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        for p in self._processors:
            p.on_start(span, parent_context=parent_context)

    def _on_ending(self, span: Span) -> None:
        for p in self._processors:
            if hasattr(p, "_on_ending"):
                p._on_ending(span)

    def on_end(self, span: ReadableSpan) -> None:
        for p in self._processors:
            try:
                p.on_end(span)
            except Exception:  # pragma: no cover - never break the app
                logger.debug("Span processor failed", exc_info=True)

    def shutdown(self) -> None:
        for p in self._processors:
            p.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return all(p.force_flush(timeout_millis) for p in self._processors)


class _Delegating:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        self._delegate.on_start(span, parent_context=parent_context)

    def _on_ending(self, span: Span) -> None:
        self._delegate._on_ending(span)

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)


class DropSpansProcessor(_Delegating):
    """Drop leaf spans whose name matches any fnmatch pattern.

    Only safe for leaves: dropping a span with children would orphan them.
    The defaults (``*.get_metadata``, ``*._prepare_chat_with_tools``) are
    leaves emitted by llama-index auto-instrumentation.
    """

    def __init__(self, patterns: list[str], delegate: Any) -> None:
        super().__init__(delegate)
        self._regex = (
            re.compile("|".join(fnmatch.translate(p) for p in patterns))
            if patterns
            else None
        )

    def on_end(self, span: ReadableSpan) -> None:
        if self._regex is not None and self._regex.match(span.name):
            return
        self._delegate.on_end(span)


class SemconvSpanProcessor(_Delegating):
    """Normalize attributes and optionally add OTel GenAI (``gen_ai.*``) ones.

    * Replaces ``<FunctionTool object at 0x..>`` tool schemas with the JSON
      schema registered by the parent span (see :func:`register_llm_tools`),
      or drops them when unknown.
    * ``semconv="genai"|"both"`` derives ``gen_ai.*`` from OpenInference using
      OpenInference's own converter; ``genai`` then strips the OpenInference
      message payloads so content is not exported twice.
    """

    def __init__(self, semconv: str, delegate: Any) -> None:
        super().__init__(delegate)
        self._semconv = semconv
        self._to_genai = None
        if semconv != "openinference":
            try:
                from openinference.instrumentation._genai_conversion import (
                    get_genai_attributes,
                )

                self._to_genai = get_genai_attributes
            except ImportError:
                logger.warning(
                    "observability.semconv=%s needs openinference-instrumentation"
                    ">=0.1.71; exporting OpenInference attributes only",
                    semconv,
                )

    def _fix_tools(self, span: ReadableSpan, attrs: dict[str, Any]) -> bool:
        broken = [
            (key, int(m.group(1)))
            for key, value in attrs.items()
            if (m := _TOOL_SCHEMA_KEY.match(key))
            and isinstance(value, str)
            and _BROKEN_REPR.match(value)
        ]
        if not broken:
            return False
        registered: list[str] = []
        if span.parent is not None:
            with _tools_lock:
                registered = _tools_by_parent.get(span.parent.span_id, [])
        for key, index in broken:
            if index < len(registered):
                attrs[key] = registered[index]
            else:
                del attrs[key]
        return True

    def on_end(self, span: ReadableSpan) -> None:
        attrs = dict(span.attributes or {})
        changed = self._fix_tools(span, attrs)
        if self._to_genai is not None and attrs.get("openinference.span.kind"):
            try:
                genai = self._to_genai(cast("Any", attrs))
            except Exception:  # pragma: no cover - converter is best effort
                logger.debug("gen_ai conversion failed", exc_info=True)
                genai = {}
            if genai:
                attrs.update(genai)
                changed = True
            if self._semconv == "genai":
                for key in [
                    k
                    for k in attrs
                    if k.startswith(("llm.input_messages.", "llm.output_messages."))
                ]:
                    del attrs[key]
        self._delegate.on_end(_with_attributes(span, attrs) if changed else span)
