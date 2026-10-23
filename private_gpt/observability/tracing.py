"""Thin, dependency-optional tracing helpers used across the codebase.

Only ``opentelemetry-api`` is needed at import time; without it every helper
is a no-op. Span attributes follow OpenInference names; the
``SemconvSpanProcessor`` adds the OTel GenAI equivalents at export time when
``observability.semconv`` asks for them, so call sites never write two
conventions.
"""

from __future__ import annotations

import functools
import json
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence

F = TypeVar("F", bound="Callable[..., Any]")

logger = logging.getLogger(__name__)

try:
    from opentelemetry import context as otel_context
    from opentelemetry import propagate, trace
    from opentelemetry.trace import Link, Status, StatusCode

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _OTEL_AVAILABLE = False

TRACER_NAME = "private_gpt"

# OpenInference attribute keys (kept local so the semconv package is optional).
SPAN_KIND = "openinference.span.kind"
INPUT_VALUE = "input.value"
INPUT_MIME_TYPE = "input.mime_type"
OUTPUT_VALUE = "output.value"
OUTPUT_MIME_TYPE = "output.mime_type"
SESSION_ID = "session.id"
USER_ID = "user.id"
METADATA = "metadata"
TAGS = "tag.tags"
TOOL_NAME = "tool.name"
TOOL_DESCRIPTION = "tool.description"
TOOL_PARAMETERS = "tool.parameters"
TOOL_CALL_ID = "tool_call.id"
AGENT_NAME = "agent.name"
LLM_MODEL_NAME = "llm.model_name"
LLM_PROVIDER = "llm.provider"
LLM_SYSTEM = "llm.system"
TOKEN_PROMPT = "llm.token_count.prompt"
TOKEN_COMPLETION = "llm.token_count.completion"
TOKEN_TOTAL = "llm.token_count.total"
TOKEN_CACHE_READ = "llm.token_count.prompt_details.cache_read"
TOKEN_REASONING = "llm.token_count.completion_details.reasoning"
GRAPH_NODE_ID = "graph.node.id"
GRAPH_NODE_NAME = "graph.node.name"
GRAPH_NODE_PARENT_ID = "graph.node.parent_id"

TRACE_CONTEXT_KEY = "_trace"

_MAX_ATTRIBUTE_CHARS = 32_000


class SpanKind(StrEnum):
    """OpenInference span kinds (rendered natively by most LLM backends)."""

    AGENT = "AGENT"
    CHAIN = "CHAIN"
    TOOL = "TOOL"
    LLM = "LLM"
    RETRIEVER = "RETRIEVER"
    RERANKER = "RERANKER"
    EMBEDDING = "EMBEDDING"
    GUARDRAIL = "GUARDRAIL"
    EVALUATOR = "EVALUATOR"


def is_enabled() -> bool:
    if not _OTEL_AVAILABLE:
        return False
    from private_gpt.observability import setup

    return setup.is_configured()


def _serialize(value: Any) -> tuple[str, str]:
    if isinstance(value, str):
        text, mime = value, "text/plain"
    else:
        try:
            text = json.dumps(value, default=str, ensure_ascii=False)
            mime = "application/json"
        except (TypeError, ValueError):
            text, mime = str(value), "text/plain"
    if len(text) > _MAX_ATTRIBUTE_CHARS:
        text = text[:_MAX_ATTRIBUTE_CHARS] + "…[truncated]"
    return text, mime


def _clean(attributes: Mapping[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            cleaned[key] = value
        elif isinstance(value, (list, tuple)) and all(
            isinstance(v, (str, bool, int, float)) for v in value
        ):
            cleaned[key] = list(value)
        else:
            cleaned[key] = _serialize(value)[0]
    return cleaned


def set_span_attributes(span: Any = None, /, **attributes: Any) -> None:
    """Set attributes on ``span`` (default: current span). ``None`` is skipped.

    Use dotted keys through ``**{"llm.model_name": ...}`` or the module
    constants. ``input``/``output`` are serialized to ``input.value``/
    ``output.value`` with the matching mime type unless content capture is off.
    """
    if not _OTEL_AVAILABLE:
        return
    span = span or trace.get_current_span()
    if not span.is_recording():
        return
    from private_gpt.observability import setup

    for io_key in ("input", "output"):
        if io_key in attributes:
            value = attributes.pop(io_key)
            if value is not None and setup.capture_content():
                text, mime = _serialize(value)
                attributes[f"{io_key}.value"] = text
                attributes[f"{io_key}.mime_type"] = mime
    span.set_attributes(_clean(attributes))


def set_span_status(span: Any = None, *, error: str | None = None) -> None:
    """Mark ``span`` as failed (``error`` given) or successful.

    The first final status wins. The SDK lets a later OK overwrite an ERROR
    (and freezes a span once it is OK), so an ERROR is never replaced here:
    a span failed by an outcome stays failed when its ``start_span`` closes.
    """
    if not _OTEL_AVAILABLE:
        return
    span = span or trace.get_current_span()
    if not span.is_recording():
        return
    current = getattr(getattr(span, "status", None), "status_code", None)
    if current is StatusCode.ERROR:
        return
    if error is None:
        span.set_status(Status(StatusCode.OK))
    else:
        span.set_status(Status(StatusCode.ERROR, error[:500]))


def record_exception(exc: BaseException, span: Any = None) -> None:
    if not _OTEL_AVAILABLE:
        return
    import asyncio

    span = span or trace.get_current_span()
    if not span.is_recording():
        return
    if isinstance(exc, asyncio.CancelledError):
        # Cancellation (client disconnect, user stop) is not a failure.
        span.set_attribute("pgpt.cancelled", True)
        return
    span.record_exception(exc)
    set_span_status(span, error=f"{type(exc).__name__}: {exc}")


@contextmanager
def start_span(
    name: str,
    kind: SpanKind | str = SpanKind.CHAIN,
    *,
    attributes: Mapping[str, Any] | None = None,
    input: Any = None,
    links: list[Any] | None = None,
    context: Any = None,
) -> Iterator[Any]:
    """Start an OpenInference-typed span as the current span.

    A span that exits normally ends with status OK unless something marked it
    as failed (``set_span_status(error=...)``) while it was open. Exceptions
    are recorded as ERROR and re-raised; ``asyncio.CancelledError`` is tagged
    as cancelled and left UNSET, since a stopped turn did not fail.
    """
    if not is_enabled():
        yield _NOOP_SPAN
        return
    tracer = trace.get_tracer(TRACER_NAME)
    if context is None:
        context = _llama_index_parent()
    base = {SPAN_KIND: str(kind)}
    # OpenInference context attributes (session/user/metadata/tags) are copied
    # onto every span so that each backend can group without walking parents.
    base.update(_context_attributes())
    if attributes:
        base.update(_clean(attributes))
    with tracer.start_as_current_span(
        name,
        context=context,
        attributes=base,
        links=links,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        if input is not None:
            set_span_attributes(span, input=input)
        try:
            yield span
        except BaseException as exc:
            record_exception(exc, span)
            raise
        set_span_status(span)


def _llama_index_parent() -> Any:
    """Context of the innermost open llama-index span, when it is deeper.

    llama-index workflow steps run in their own tasks: they know their
    llama-index span (``active_span_id``) but the OTel current span is still
    the one that started the workflow. Spans we open inside a step
    (``tabular.analysis``, ``web_search.query``...) would otherwise become
    siblings of the workflow instead of children of the step.
    """
    try:
        from llama_index_instrumentation.span import active_span_id
        from openinference.instrumentation.llama_index import _handler
    except ImportError:
        return None
    li_span = _open_llama_index_span(active_span_id, _handler)
    if li_span is None:
        return None
    current = trace.get_current_span().get_span_context()
    li_ctx = li_span._otel_span.get_span_context()
    if current.is_valid and current.trace_id != li_ctx.trace_id:
        return None
    current_start = getattr(trace.get_current_span(), "start_time", 0) or 0
    if (getattr(li_span._otel_span, "start_time", 0) or 0) < current_start:
        return None
    return li_span.context


def _open_llama_index_span(active_span_id: Any, module: Any) -> Any:
    span_id = active_span_id.get()
    if not span_id:
        return None
    handler = _llama_index_span_handler(module)
    if handler is None:
        return None
    with handler.lock:
        li_span = handler.open_spans.get(span_id)
    if li_span is None or not li_span.active:
        return None
    return li_span


# Set by an LLM backend while it builds a request; read by the llama-index
# LLM span when its end event arrives (see llama_index_patch). Generators run
# in the consumer's context, so a value set inside a streaming generator is
# visible to the end event dispatched after it.
pending_invocation_parameters: ContextVar[str | None] = ContextVar(
    "pgpt_invocation_parameters", default=None
)


def set_llm_invocation_parameters(parameters: Mapping[str, Any]) -> None:
    """Record the parameters an LLM request is actually sent with.

    The llama-index instrumentation only records ``LLM.metadata``; backends
    that resolve their parameters per call (sampling, dynamic ``max_tokens``,
    structured outputs...) call this right before sending the request, and
    the LLM span shows them as ``llm.invocation_parameters``.
    """
    if not is_enabled():
        return
    pending_invocation_parameters.set(
        json.dumps(parameters, default=str, sort_keys=True)
    )


_li_handler: Any = None


def _llama_index_span_handler(module: Any) -> Any:
    global _li_handler
    from llama_index_instrumentation import get_dispatcher

    handlers = get_dispatcher().span_handlers
    # Re-instrumenting replaces the handler; never keep a detached one.
    if _li_handler is None or _li_handler not in handlers:
        _li_handler = next(
            (h for h in handlers if isinstance(h, module._SpanHandler)), None
        )
    return _li_handler


def retrieval_documents(documents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """OpenInference ``retrieval.documents.N.document.*`` attributes.

    Each document may carry ``id``, ``content``, ``score`` and ``metadata``;
    backends render them as the RETRIEVER span's result list.
    """
    attributes: dict[str, Any] = {}
    for index, document in enumerate(documents):
        prefix = f"retrieval.documents.{index}.document"
        for key in ("id", "content", "score"):
            if document.get(key) is not None:
                attributes[f"{prefix}.{key}"] = document[key]
        if document.get("metadata"):
            attributes[f"{prefix}.metadata"] = _serialize(document["metadata"])[0]
    return attributes


def traced(
    name: str | None = None,
    kind: SpanKind | str = SpanKind.CHAIN,
    *,
    capture: Callable[..., Any] | None = None,
    result: Callable[[Any], Any] | None = None,
) -> Callable[[F], F]:
    """Decorate an async function so each call runs inside ``start_span``.

    ``capture(*args, **kwargs)`` picks the span input (default: keyword
    arguments, without ``self``); ``result(value)`` shapes the span output
    (default: the returned value, serialized). Both run only when tracing is
    enabled and never break the call.
    """

    def decorate(fn: F) -> F:
        span_name = name or getattr(fn, "__qualname__", repr(fn))

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_enabled():
                return await fn(*args, **kwargs)
            try:
                captured = capture(*args, **kwargs) if capture else kwargs or None
            except Exception:
                captured = None
            with start_span(span_name, kind, input=captured) as span:
                value = await fn(*args, **kwargs)
                try:
                    set_span_attributes(span, output=result(value) if result else value)
                except Exception:
                    logger.debug("Could not record output of %s", span_name)
                return value

        return cast("F", wrapper)

    return decorate


def _context_attributes() -> dict[str, Any]:
    try:
        from openinference.instrumentation import get_attributes_from_context
    except ImportError:
        return {}
    return dict(get_attributes_from_context())


@contextmanager
def using_trace_attributes(
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[None]:
    """Attach session/user/metadata/tags to every span started in this block.

    Uses OpenInference context attributes, so llama-index spans created by the
    OpenInference instrumentor inherit them as well.
    """
    try:
        from openinference.instrumentation import using_attributes
    except ImportError:
        yield
        return
    kwargs: dict[str, Any] = {}
    if session_id:
        kwargs["session_id"] = session_id
    if user_id:
        kwargs["user_id"] = user_id
    if metadata:
        kwargs["metadata"] = {k: v for k, v in metadata.items() if v is not None}
    if tags:
        kwargs["tags"] = tags
    with using_attributes(**kwargs):
        yield


def current_trace_id() -> str | None:
    """Hex trace id of the active span, or ``None`` when not tracing."""
    if not _OTEL_AVAILABLE:
        return None
    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return None
    return f"{ctx.trace_id:032x}"


def inject_trace_context(carrier: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return ``carrier`` with W3C ``traceparent``/``tracestate``/baggage.

    Also serializes the OpenInference context attributes so a worker restores
    session/user grouping, not just the parent span.
    """
    carrier = {} if carrier is None else carrier
    if not _OTEL_AVAILABLE:
        return carrier
    propagate.inject(carrier)
    oi_attributes = _context_attributes()
    if oi_attributes:
        carrier["oi"] = oi_attributes
    return carrier


@contextmanager
def trace_context_from(carrier: Mapping[str, Any] | None) -> Iterator[None]:
    """Attach a context extracted by :func:`inject_trace_context`."""
    if not _OTEL_AVAILABLE or not carrier:
        yield
        return
    ctx = propagate.extract(dict(carrier))
    for key, value in (carrier.get("oi") or {}).items():
        ctx = otel_context.set_value(key, value, ctx)
    token = otel_context.attach(ctx)
    try:
        yield
    finally:
        otel_context.detach(token)


def link_from(carrier: Mapping[str, Any] | None) -> list[Any]:
    """Span link to the span stored in ``carrier`` (for resumed executions)."""
    if not _OTEL_AVAILABLE or not carrier:
        return []
    span_ctx = trace.get_current_span(
        propagate.extract(dict(carrier))
    ).get_span_context()
    return [Link(span_ctx)] if span_ctx.is_valid else []


def trace_reference() -> dict[str, str]:
    """``trace_id``/``traceparent`` of the current span, for stored records.

    Safe to expose to clients (no session/user attributes), unlike the full
    carrier from :func:`inject_trace_context`.
    """
    if not _OTEL_AVAILABLE:
        return {}
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    trace_id = current_trace_id()
    if trace_id is None or "traceparent" not in carrier:
        return {}
    return {"trace_id": trace_id, "traceparent": carrier["traceparent"]}


def link_current_span(reference: Mapping[str, Any] | None, **attributes: Any) -> None:
    """Link the current span to a :func:`trace_reference` (e.g. a turn).

    Used by requests that observe work started by another request (async
    stream / status), so trace backends can jump to the originating trace.
    """
    if not is_enabled():
        return
    span = trace.get_current_span()
    if not span.is_recording():
        return
    pgpt = {f"pgpt.{key}": value for key, value in attributes.items()}
    if reference and reference.get("traceparent"):
        own = span.get_span_context()
        for link in link_from({"traceparent": reference["traceparent"]}):
            # `POST /v1/messages` starts the turn and observes it in the same
            # request: a link to itself makes backends lose the root.
            if link.context.span_id == own.span_id:
                continue
            span.add_link(link.context, {"pgpt.link": "origin"})
        if reference.get("trace_id") != f"{own.trace_id:032x}":
            pgpt["pgpt.origin.trace_id"] = reference.get("trace_id")
    set_span_attributes(span, **pgpt)


class _NoopSpan:
    def set_attribute(self, *_: Any, **__: Any) -> None: ...

    def set_attributes(self, *_: Any, **__: Any) -> None: ...

    def add_event(self, *_: Any, **__: Any) -> None: ...

    def set_status(self, *_: Any, **__: Any) -> None: ...

    def is_recording(self) -> bool:
        return False


_NOOP_SPAN = _NoopSpan()
