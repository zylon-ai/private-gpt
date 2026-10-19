"""Trace tags: rules tag whole traces, exporters keep only their tags."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from private_gpt.observability.exporters import exporter_tags
from private_gpt.observability.processors import (
    TRACE_TAGS,
    DropSpansProcessor,
    FanOutSpanProcessor,
    TagFilterProcessor,
    TraceTagsProcessor,
)
from private_gpt.settings.settings import (
    ObservabilityExporterSettings,
    ObservabilitySettings,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Tracer


def _pipeline(
    *exporters: tuple[list[str], InMemorySpanExporter],
    drop: list[str] | None = None,
) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(
        TraceTagsProcessor(  # ty: ignore[invalid-argument-type]
            rules=ObservabilitySettings().trace_tags,
            delegate=DropSpansProcessor(
                patterns=drop or [],
                delegate=FanOutSpanProcessor(
                    [
                        TagFilterProcessor(tags, SimpleSpanProcessor(memory))
                        for tags, memory in exporters
                    ]
                ),
            ),
        )
    )
    return provider


def _tree(tracer: Tracer, *names: str, attributes: dict | None = None) -> None:
    """Open ``names`` as nested spans (first is the root)."""
    with contextlib.ExitStack() as stack:
        for index, name in enumerate(names):
            last = index == len(names) - 1
            stack.enter_context(
                tracer.start_as_current_span(
                    name, attributes=attributes if last else None
                )
            )


def _names(memory: InMemorySpanExporter) -> list[str]:
    return sorted(span.name for span in memory.get_finished_spans())


def test_ai_trace_reaches_llm_backend_and_generic_otel() -> None:
    phoenix, otel = InMemorySpanExporter(), InMemorySpanExporter()
    tracer = _pipeline((["ai"], phoenix), (["all"], otel)).get_tracer("t")

    _tree(tracer, "POST /v1/messages", "chat.turn", "tool search")
    with tracer.start_as_current_span("GET /v1/messages/async/{message_id}/status"):
        pass
    with tracer.start_as_current_span("GET /v1/artifacts/list"):
        pass

    assert _names(phoenix) == ["POST /v1/messages", "chat.turn", "tool search"]
    assert len(otel.get_finished_spans()) == 5
    # Every span of the tree carries the tag, not just the matching root.
    assert all(
        span.attributes and list(span.attributes[TRACE_TAGS]) == ["ai"]
        for span in phoenix.get_finished_spans()
    )


def test_worker_side_of_a_trace_is_tagged_by_its_own_spans() -> None:
    """ARQ / Celery workers see only their part of the trace."""
    phoenix = InMemorySpanExporter()
    tracer = _pipeline((["ai"], phoenix)).get_tracer("t")

    _tree(tracer, "chat.turn.resume", "chat.iteration 2")
    _tree(tracer, "run/private_gpt.tools.run", "sandbox.exec")

    assert _names(phoenix) == [
        "chat.iteration 2",
        "chat.turn.resume",
        "run/private_gpt.tools.run",
        "sandbox.exec",
    ]


def test_ingestion_is_its_own_tag() -> None:
    phoenix, ingestion = InMemorySpanExporter(), InMemorySpanExporter()
    tracer = _pipeline((["ai"], phoenix), (["ingestion"], ingestion)).get_tracer("t")

    _tree(
        tracer,
        "POST /v1/artifacts/ingest/async",
        "apply_async/private_gpt.ingestion.parse",
    )
    _tree(tracer, "run/private_gpt.ingestion.store_vectors", "describe_image")

    assert phoenix.get_finished_spans() == ()
    assert len(ingestion.get_finished_spans()) == 4


def test_kind_rules_and_dropped_spans_keep_counts_balanced() -> None:
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    tagger = TraceTagsProcessor(  # ty: ignore[invalid-argument-type]
        rules={"llm": ["kind:LLM"]},
        delegate=DropSpansProcessor(
            patterns=["noise"],
            delegate=TagFilterProcessor(["llm"], SimpleSpanProcessor(memory)),
        ),
    )
    provider.add_span_processor(tagger)  # ty: ignore[invalid-argument-type]
    tracer = provider.get_tracer("t")

    _tree(
        tracer,
        "root",
        "noise",
        "llm.chat",
        attributes={"openinference.span.kind": "LLM"},
    )

    # The LLM span ends first, so it is tagged when it matches; the root ends
    # later and inherits. The dropped span is not exported.
    assert _names(memory) == ["llm.chat", "root"]
    assert tagger._traces == {}


def test_exporter_default_tags() -> None:
    assert exporter_tags(ObservabilityExporterSettings(type="phoenix")) == ["ai"]
    assert exporter_tags(ObservabilityExporterSettings(type="opik")) == ["ai"]
    assert exporter_tags(ObservabilityExporterSettings(type="otlp")) == ["all"]
    assert exporter_tags(
        ObservabilityExporterSettings(type="phoenix", tags=["ai", "ingestion"])
    ) == ["ai", "ingestion"]


def test_async_observer_spans_link_to_the_turn_trace() -> None:
    from opentelemetry import trace

    from private_gpt.observability import setup
    from private_gpt.observability.tracing import link_current_span, trace_reference

    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    setup._configured = True
    setup._settings = ObservabilitySettings(mode="otel")
    try:
        tracer = provider.get_tracer("t")
        with tracer.start_as_current_span("POST /v1/messages/async") as origin:
            reference = trace_reference()
        with tracer.start_as_current_span("GET /v1/messages/async/{id}/status"):
            link_current_span(reference, message_id="m1")
    finally:
        setup._reset_for_tests()
        trace._TRACER_PROVIDER = None  # type: ignore[attr-defined]

    origin_ctx = origin.get_span_context()
    assert reference["trace_id"] == f"{origin_ctx.trace_id:032x}"
    status = next(s for s in memory.get_finished_spans() if s.name.startswith("GET"))
    assert [link.context.span_id for link in status.links] == [origin_ctx.span_id]
    assert status.attributes
    assert status.attributes["pgpt.message_id"] == "m1"
    assert status.attributes["pgpt.origin.trace_id"] == reference["trace_id"]
