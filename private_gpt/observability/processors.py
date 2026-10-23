"""Span processors applied once, before spans fan out to every exporter."""

from __future__ import annotations

import contextlib
import fnmatch
import json
import logging
import re
import threading
from time import time_ns
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opentelemetry.context import Context
    from opentelemetry.sdk.trace import ReadableSpan, Span

logger = logging.getLogger(__name__)


SPAN_KIND = "openinference.span.kind"
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
    return _copy(span, attributes=attributes)


def _with_parent(span: ReadableSpan, parent: Any) -> ReadableSpan:
    return _copy(span, parent=parent)


def _copy(span: ReadableSpan, **changes: Any) -> ReadableSpan:
    from opentelemetry.sdk.trace import ReadableSpan

    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=changes.get("parent", span.parent),
        resource=changes.get("resource", span.resource),
        attributes=changes.get("attributes", span.attributes),
        events=span.events,
        links=span.links,
        kind=span.kind,
        status=changes.get("status", span.status),
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
        on_ending = getattr(self._delegate, "_on_ending", None)
        if on_ending is not None:
            on_ending(span)

    def shutdown(self) -> None:
        self._delegate.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)


class DropSpansProcessor(_Delegating):
    """Drop spans whose name matches any fnmatch pattern.

    Children of a dropped span are re-attached to the dropped span's parent,
    so nothing is orphaned. Children end before their parent, so the mapping
    is recorded when the dropped span starts.
    """

    def __init__(self, patterns: list[str], delegate: Any) -> None:
        super().__init__(delegate)
        self._regex = (
            re.compile("|".join(fnmatch.translate(p) for p in patterns))
            if patterns
            else None
        )
        self._lock = threading.Lock()
        # dropped span id -> its parent context
        self._reparent: dict[int, Any] = {}
        # spans with at least one exported child (children end first)
        self._has_children: set[int] = set()

    def _dropped(self, name: str) -> bool:
        return self._regex is not None and bool(self._regex.match(name))

    def _idle(self, span: ReadableSpan) -> bool:
        """An interceptor or condensation check that did no work and did not fail."""
        attrs = span.attributes or {}
        if not span.status.is_ok:
            return False
        if attrs.get("pgpt.condensation.condensed") is False:
            return True
        return (
            attrs.get("pgpt.interceptor.changed") is False
            and span.context.span_id not in self._has_children
        )

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        if self._dropped(span.name):
            with self._lock:
                self._reparent[span.context.span_id] = self._resolve(span.parent)
            return
        super().on_start(span, parent_context=parent_context)

    def _resolve(self, parent: Any) -> Any:
        while parent is not None and parent.span_id in self._reparent:
            parent = self._reparent[parent.span_id]
        return parent

    def on_end(self, span: ReadableSpan) -> None:
        span_id = span.context.span_id
        if self._dropped(span.name):
            with self._lock:
                self._reparent.pop(span_id, None)
            return
        with self._lock:
            idle = self._idle(span)
            self._has_children.discard(span_id)
        if idle:
            return
        if span.parent is not None:
            with self._lock:
                if span.parent.span_id in self._reparent:
                    span = _with_parent(span, self._resolve(span.parent))
                if span.parent is not None:
                    self._has_children.add(span.parent.span_id)
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

    _TOKEN_KEYS = (
        "llm.token_count.prompt",
        "llm.token_count.completion",
        "llm.token_count.total",
    )

    def __init__(self, semconv: str, delegate: Any) -> None:
        super().__init__(delegate)
        self._semconv = semconv
        # LLM span id -> token counts of a nested LLM call (children end
        # first); LLM wrappers like Triton's chat -> complete only get usage
        # on the inner span.
        self._child_tokens: dict[int, dict[str, Any]] = {}
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

    @staticmethod
    def _celery(span: ReadableSpan, attrs: dict[str, Any]) -> Any:
        """Give Celery task spans a kind and, for runs, an OK/ERROR status.

        The Celery instrumentation leaves them kind-less (``unknown`` in LLM
        backends) and only sets ERROR, never OK; ``celery.state`` has the
        outcome. Returns the status to apply, or ``None``.
        """
        if "celery.action" not in attrs:
            return None
        attrs.setdefault("openinference.span.kind", "CHAIN")
        if not span.status.is_unset:
            return None
        from opentelemetry.trace import Status, StatusCode

        state = attrs.get("celery.state")
        if state in ("FAILURE", "REVOKED"):
            return Status(StatusCode.ERROR, f"Celery task {state}")
        if state == "SUCCESS" or attrs.get("celery.action") == "apply_async":
            return Status(StatusCode.OK)
        return None

    def _carry_tokens(self, span: ReadableSpan, attrs: dict[str, Any]) -> bool:
        if attrs.get(SPAN_KIND) != "LLM":
            return False
        inherited = self._child_tokens.pop(span.context.span_id, None)
        changed = False
        if inherited and not any(key in attrs for key in self._TOKEN_KEYS):
            attrs.update(inherited)
            changed = True
        tokens = {key: attrs[key] for key in self._TOKEN_KEYS if key in attrs}
        if tokens and span.parent is not None:
            self._child_tokens[span.parent.span_id] = tokens
            if len(self._child_tokens) > 10_000:  # parents that never ended
                self._child_tokens.pop(next(iter(self._child_tokens)))
        return changed

    def on_end(self, span: ReadableSpan) -> None:
        attrs = dict(span.attributes or {})
        changed = self._fix_tools(span, attrs)
        changed = self._carry_tokens(span, attrs) or changed
        celery_kind = "celery.action" in attrs and SPAN_KIND not in attrs
        status = self._celery(span, attrs)
        changed = changed or celery_kind
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
        if status is not None:
            span = _copy(span, attributes=attrs, status=status)
        elif changed:
            span = _with_attributes(span, attrs)
        self._delegate.on_end(span)


TRACE_TAGS = "pgpt.trace.tags"
_OI_TAGS = "tag.tags"
ALL_TAGS = "all"


class TraceTagsProcessor(_Delegating):
    """Tag whole traces from span-name rules (``observability.trace_tags``).

    A rule is an fnmatch pattern on the span name (``POST /v1/messages``,
    ``chat.turn*``, ``run/private_gpt.ingestion.parse``) or ``kind:<KIND>`` on
    the OpenInference span kind. When any span of a trace matches, every span
    of that trace still open or ending later in this process carries the tag
    in ``pgpt.trace.tags`` (and OpenInference ``tag.tags``), so exporters can
    keep or skip whole trees (see :class:`TagFilterProcessor`).

    Matching is per process: a worker continuing a trace re-derives the tags
    from its own spans (``chat.turn``, Celery ``run/<task>``), which the
    default rules cover.
    """

    def __init__(self, rules: Mapping[str, list[str]], delegate: Any) -> None:
        super().__init__(delegate)
        self._names: list[tuple[str, re.Pattern[str]]] = []
        self._kinds: list[tuple[str, set[str]]] = []
        for tag, patterns in rules.items():
            names = [p for p in patterns if not p.startswith("kind:")]
            kinds = {p.removeprefix("kind:") for p in patterns if p.startswith("kind:")}
            if names:
                self._names.append(
                    (tag, re.compile("|".join(fnmatch.translate(p) for p in names)))
                )
            if kinds:
                self._kinds.append((tag, kinds))
        self._lock = threading.Lock()
        # trace id -> [open spans, tags]
        self._traces: dict[int, list[Any]] = {}

    def _match(self, span: ReadableSpan) -> set[str]:
        tags = {tag for tag, regex in self._names if regex.match(span.name)}
        kind = (span.attributes or {}).get("openinference.span.kind")
        if kind:
            tags.update(tag for tag, kinds in self._kinds if kind in kinds)
        return tags

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        tags = self._match(span)
        with self._lock:
            entry = self._traces.setdefault(span.context.trace_id, [0, set()])
            entry[0] += 1
            entry[1].update(tags)
        super().on_start(span, parent_context=parent_context)

    def on_end(self, span: ReadableSpan) -> None:
        trace_id = span.context.trace_id
        tags = self._match(span)
        with self._lock:
            entry = self._traces.get(trace_id)
            if entry is not None:
                entry[1].update(tags)
                tags = set(entry[1])
                entry[0] -= 1
                if entry[0] <= 0:
                    del self._traces[trace_id]
        if tags:
            attrs = dict(span.attributes or {})
            attrs[TRACE_TAGS] = sorted(tags)
            existing = attrs.get(_OI_TAGS)
            if isinstance(existing, (list, tuple)):
                tags = tags.union(str(tag) for tag in existing)
            attrs[_OI_TAGS] = sorted(tags)
            span = _with_attributes(span, attrs)
        self._delegate.on_end(span)


class TagFilterProcessor(_Delegating):
    """Forward only spans whose trace carries one of ``tags`` (``all``: every span)."""

    def __init__(self, tags: list[str], delegate: Any) -> None:
        super().__init__(delegate)
        self._all = ALL_TAGS in tags
        self._tags = set(tags)

    def accepts(self, span: ReadableSpan) -> bool:
        if self._all:
            return True
        tags = (span.attributes or {}).get(TRACE_TAGS)
        return isinstance(tags, (list, tuple)) and not self._tags.isdisjoint(
            str(tag) for tag in tags
        )

    def on_end(self, span: ReadableSpan) -> None:
        if self.accepts(span):
            self._delegate.on_end(span)


class ExporterViewProcessor(_Delegating):
    """Per-exporter shape of a trace: project and the chat-turn view.

    * ``resource``: attributes merged into each span's resource, so two
      exporters of the same backend can write to different projects
      (e.g. ``openinference.project.name``).
    * ``turn_root``: the HTTP span of a chat route stays the root of the
      trace, with ``chat.turn`` under it. Its input is the JSON request; its
      output would be the SSE text or, for ``/messages/async``, the
      ``{"message_id": ...}`` acknowledgement. So the turn's final message
      (JSON, as a non-streaming response would return it), session and user
      are copied onto the HTTP span (the raw response body moves to
      ``pgpt.http.response.body``); the folded user input and the answer text
      are on ``chat.turn``. An async POST ends before its turn: it is held until
      the turn ends, or exported as is after ``hold_seconds`` (turn in another
      process, e.g. an ARQ worker).
    """

    _TURN_KEYS = (
        "output.value",
        "output.mime_type",
        "session.id",
        "user.id",
        "metadata",
        "pgpt.output.message",
    )

    def __init__(
        self,
        delegate: Any,
        *,
        resource: Mapping[str, str] | None = None,
        turn_root: bool = False,
        turn_routes: list[str] | None = None,
        hold_seconds: float = 600,
    ) -> None:
        super().__init__(delegate)
        self._resource_attributes = dict(resource or {})
        self._resources: dict[int, Any] = {}
        self._turn_root = turn_root
        self._routes = (
            re.compile("|".join(fnmatch.translate(p) for p in turn_routes))
            if turn_routes
            else None
        )
        self._hold_ns = int(hold_seconds * 10**9)
        self._lock = threading.Lock()
        # HTTP span id of a chat route -> attributes of its finished turn
        self._turns: dict[int, dict[str, Any]] = {}
        # ended HTTP spans waiting for their turn: span id -> span
        self._held: dict[int, ReadableSpan] = {}

    def _resource(self, span: ReadableSpan) -> Any:
        if not self._resource_attributes:
            return span.resource
        key = id(span.resource)
        resource = self._resources.get(key)
        if resource is None:
            resource = span.resource.merge(
                type(span.resource)(self._resource_attributes)
            )
            self._resources[key] = resource
        return resource

    def _is_turn_route(self, span: Any) -> bool:
        from opentelemetry.trace import SpanKind as OtelKind

        return (
            self._routes is not None
            and span.kind is OtelKind.SERVER
            and bool(self._routes.match(span.name))
        )

    def _with_turn(self, span: ReadableSpan, turn: Mapping[str, Any]) -> ReadableSpan:
        attrs = dict(span.attributes or {})
        if "output.value" in attrs:
            attrs["pgpt.http.response.body"] = attrs["output.value"]
        attrs.update(turn)
        message = turn.get("pgpt.output.message")
        if isinstance(message, str) and message:
            attrs["output.value"] = message
            attrs["output.mime_type"] = "application/json"
        return _with_attributes(span, attrs)

    def _export(self, span: ReadableSpan) -> None:
        if self._resource_attributes:
            span = _copy(span, resource=self._resource(span))
        self._delegate.on_end(span)

    def _expired(self, now: int) -> list[ReadableSpan]:
        expired = [
            span_id
            for span_id, span in self._held.items()
            if (span.end_time or 0) < now - self._hold_ns
        ]
        return [self._held.pop(span_id) for span_id in expired]

    def on_end(self, span: ReadableSpan) -> None:
        if not self._turn_root:
            self._export(span)
            return
        ready: list[ReadableSpan] = []
        span_id = span.context.span_id
        current: ReadableSpan | None = span
        with self._lock:
            if span.name == "chat.turn" and span.parent is not None:
                attrs = span.attributes or {}
                turn = {k: attrs[k] for k in self._TURN_KEYS if k in attrs}
                parent_id = span.parent.span_id
                held = self._held.pop(parent_id, None)
                if held is not None:
                    ready.append(self._with_turn(held, turn))
                else:
                    self._turns[parent_id] = turn
            if self._is_turn_route(span):
                turn = self._turns.pop(span_id, None)
                if turn is not None:
                    current = self._with_turn(span, turn)
                elif span.status.status_code.name != "ERROR":
                    # Async: the turn has not run yet. (A failed request
                    # never had one; it is exported as is.)
                    self._held[span_id] = span
                    current = None
            ready.extend(self._expired(time_ns()))
        if current is not None:
            ready.append(current)
        for item in ready:
            self._export(item)

    def shutdown(self) -> None:
        with self._lock:
            held, self._held = list(self._held.values()), {}
        for span in held:
            self._export(span)
        super().shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)


class BackgroundSpanProcessor(_Delegating):
    """Run the processing chain off the request path.

    ``on_start`` stays inline (it only records ids, and children need it
    before they start); ``on_end`` only enqueues the finished span, and a
    daemon thread runs the rest of the chain (tagging, dropping, attribute
    normalization, per-exporter filtering) and hands spans to the exporters'
    own batch queues. When the queue is full spans are dropped and counted,
    never waited on, so a slow or unreachable backend cannot slow requests.
    """

    def __init__(self, delegate: Any, max_queue_size: int = 8192) -> None:
        super().__init__(delegate)
        import queue

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
        self.dropped = 0
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="pgpt-span-processor", daemon=True
        )
        self._thread.start()

    def on_end(self, span: ReadableSpan) -> None:
        if self._closed:
            return
        try:
            self._queue.put_nowait(span)
        except Exception:
            self.dropped += 1
            if self.dropped == 1 or self.dropped % 1000 == 0:
                logger.warning("Span queue full; %d spans dropped", self.dropped)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if isinstance(item, threading.Event):
                    item.set()
                    continue
                self._delegate.on_end(item)
            except Exception:  # pragma: no cover - never kill the worker
                logger.debug("Span processing failed", exc_info=True)
            finally:
                self._queue.task_done()

    def _drain(self, timeout_millis: int) -> bool:
        marker = threading.Event()
        try:
            self._queue.put(marker, timeout=timeout_millis / 1000)
        except Exception:
            return False
        return marker.wait(timeout_millis / 1000)

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._drain(timeout_millis) and self._delegate.force_flush(
            timeout_millis
        )

    def shutdown(self) -> None:
        # Bounded: a dead backend must not hold up process exit (Celery
        # recycles children after N tasks).
        self._drain(2000)
        self._closed = True
        with contextlib.suppress(Exception):
            self._queue.put_nowait(None)
        self._delegate.shutdown()
