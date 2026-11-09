"""Trace exporters: built-in local ones and a registry for the rest.

private-gpt ships ``jsonl`` (OTLP JSON lines to a rotating file) and
``console``. Other destinations (an OTLP endpoint, vendor backends) are
registered by the application with :func:`register_exporter`; the factory
receives the exporter settings and returns an :class:`ExporterSpec`.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from private_gpt.settings.settings import ObservabilityExporterSettings

logger = logging.getLogger(__name__)

TURN_ROUTES = ["POST /v1/messages", "POST /v1/messages/async"]


@dataclass
class ExporterSpec:
    """What a factory builds for one exporter entry.

    ``exporter`` is an OTel ``SpanExporter``. ``resource`` is merged into the
    spans' resource for this exporter only (e.g. a project name).
    ``default_tags`` apply when the entry sets no ``tags``; ``turn_root``
    copies each chat turn's result onto its HTTP root span (see
    :class:`~private_gpt.observability.processors.ExporterViewProcessor`),
    what session views of LLM backends need.
    """

    exporter: Any
    resource: dict[str, str] = field(default_factory=dict)
    default_tags: list[str] = field(default_factory=lambda: ["all"])
    turn_root: bool = False


ExporterFactory = Callable[["ObservabilityExporterSettings"], ExporterSpec]

_FACTORIES: dict[str, ExporterFactory] = {}


def register_exporter(type_name: str, factory: ExporterFactory) -> None:
    """Make ``type_name`` usable as ``observability.exporters[].type``."""
    _FACTORIES[type_name] = factory


def registered_exporters() -> list[str]:
    return sorted(_FACTORIES)


def exporter_tags(cfg: ObservabilityExporterSettings, spec: ExporterSpec) -> list[str]:
    """Trace tags ``cfg`` receives (``all`` for every span)."""
    return cfg.tags if cfg.tags is not None else spec.default_tags


class JsonlSpanExporter:
    """OTLP JSON lines: one ``ExportTraceServiceRequest`` per line.

    The format of the OpenTelemetry Collector file exporter, so the files can
    be replayed into any backend with the Collector ``otlpjsonfile`` receiver,
    or read with ``jq``. Rotated by size; never raises into the pipeline.
    """

    def __init__(self, path: str, max_bytes: int, backup_count: int) -> None:
        self._path = Path(path)
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._lock = threading.Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[Any]) -> Any:
        from google.protobuf.json_format import MessageToDict
        from opentelemetry.exporter.otlp.proto.common.trace_encoder import (
            encode_spans,
        )
        from opentelemetry.sdk.trace.export import SpanExportResult

        try:
            request = MessageToDict(encode_spans(spans))
            _hex_ids(request)
            line = json.dumps(request, separators=(",", ":"))
            with self._lock:
                self._rotate(len(line) + 1)
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception:
            logger.warning("Could not write traces to %s", self._path, exc_info=True)
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def _rotate(self, incoming: int) -> None:
        if self._max_bytes <= 0 or not self._path.exists():
            return
        if self._path.stat().st_size + incoming <= self._max_bytes:
            return
        if self._backup_count <= 0:
            self._path.unlink()
            return
        for index in range(self._backup_count - 1, 0, -1):
            older = self._path.with_name(f"{self._path.name}.{index}")
            if older.exists():
                older.replace(self._path.with_name(f"{self._path.name}.{index + 1}"))
        self._path.replace(self._path.with_name(f"{self._path.name}.1"))

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        del timeout_millis
        return True


_ID_FIELDS = ("traceId", "spanId", "parentSpanId")


def _hex_ids(request: dict[str, Any]) -> None:
    """OTLP JSON encodes ids as hex, not protobuf JSON's base64."""
    for resource_spans in request.get("resourceSpans", []):
        for scope_spans in resource_spans.get("scopeSpans", []):
            for span in scope_spans.get("spans", []):
                for item in (span, *span.get("links", [])):
                    for key in _ID_FIELDS:
                        if value := item.get(key):
                            item[key] = base64.b64decode(value).hex()


def _jsonl(cfg: ObservabilityExporterSettings) -> ExporterSpec:
    return ExporterSpec(JsonlSpanExporter(cfg.path, cfg.max_bytes, cfg.backup_count))


def _console(cfg: ObservabilityExporterSettings) -> ExporterSpec:
    del cfg
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter

    return ExporterSpec(ConsoleSpanExporter())


register_exporter("jsonl", _jsonl)
register_exporter("console", _console)


def build_span_processors(
    exporters: list[ObservabilityExporterSettings],
) -> list[Any]:
    """One ``BatchSpanProcessor`` per enabled exporter.

    Exports run in the SDK's own thread; a full queue drops spans instead of
    waiting, and the SDK restarts its worker thread after ``fork`` (Celery
    prefork).
    """
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from private_gpt.observability.processors import (
        ExporterViewProcessor,
        TagFilterProcessor,
    )

    processors: list[Any] = []
    for cfg in exporters:
        if not cfg.enabled:
            continue
        factory = _FACTORIES.get(cfg.type)
        if factory is None:
            logger.error(
                "Unknown trace exporter type %r (available: %s); skipping",
                cfg.type,
                ", ".join(registered_exporters()),
            )
            continue
        try:
            spec = factory(cfg)
        except Exception:
            logger.exception("Could not build %s trace exporter; skipping", cfg.type)
            continue
        processors.append(
            TagFilterProcessor(
                exporter_tags(cfg, spec),
                ExporterViewProcessor(
                    # Small batches keep each encode short: it holds the GIL,
                    # so a 512-span batch of LLM spans would pause the event
                    # loop ~60 ms.
                    BatchSpanProcessor(
                        spec.exporter,
                        max_queue_size=4096,
                        max_export_batch_size=32,
                        schedule_delay_millis=1000,
                        export_timeout_millis=5000,
                    ),
                    resource=spec.resource,
                    turn_root=spec.turn_root,
                    turn_routes=TURN_ROUTES,
                ),
            )
        )
    return processors
