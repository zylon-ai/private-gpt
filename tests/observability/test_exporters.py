"""Built-in exporters and the exporter registry."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from private_gpt.observability.exporters import (
    ExporterSpec,
    JsonlSpanExporter,
    build_span_processors,
    register_exporter,
)
from private_gpt.settings.settings import ObservabilityExporterSettings

if TYPE_CHECKING:
    from pathlib import Path


def _spans(count: int) -> list:
    memory = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    tracer = provider.get_tracer("t")
    for index in range(count):
        with tracer.start_as_current_span(f"span {index}", attributes={"i": index}):
            pass
    return list(memory.get_finished_spans())


def test_jsonl_writes_one_otlp_request_per_line(tmp_path: Path) -> None:
    path = tmp_path / "traces" / "traces.jsonl"
    exporter = JsonlSpanExporter(str(path), max_bytes=0, backup_count=0)

    exporter.export(_spans(2))
    exporter.export(_spans(1))

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    request = json.loads(lines[0])
    spans = request["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert [s["name"] for s in spans] == ["span 0", "span 1"]
    assert spans[1]["attributes"] == [{"key": "i", "value": {"intValue": "1"}}]
    # OTLP JSON: hex ids (protobuf JSON would give base64).
    original = _spans(1)[0]
    exporter.export([original])
    written = json.loads(path.read_text().splitlines()[-1])
    span = written["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    assert span["traceId"] == f"{original.context.trace_id:032x}"
    assert span["spanId"] == f"{original.context.span_id:016x}"


def test_jsonl_rotates_by_size(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    exporter = JsonlSpanExporter(str(path), max_bytes=1, backup_count=2)

    for _ in range(4):
        exporter.export(_spans(1))

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "traces.jsonl",
        "traces.jsonl.1",
        "traces.jsonl.2",
    ]
    assert len(path.read_text().splitlines()) == 1


def test_registered_exporter_type_gets_its_options() -> None:
    received: list[ObservabilityExporterSettings] = []
    memory = InMemorySpanExporter()

    def factory(cfg: ObservabilityExporterSettings) -> ExporterSpec:
        received.append(cfg)
        return ExporterSpec(memory, default_tags=["ai"])

    register_exporter("test-backend", factory)
    cfg = ObservabilityExporterSettings(
        type="test-backend", options={"endpoint": "http://x"}
    )

    processors = build_span_processors(
        [cfg, ObservabilityExporterSettings(type="not-registered")]
    )

    assert len(processors) == 1  # the unknown type is skipped, not fatal
    assert received[0].options == {"endpoint": "http://x"}
    assert processors[0]._tags == {"ai"}
