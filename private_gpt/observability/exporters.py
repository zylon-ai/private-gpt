"""Exporter presets. Every backend is reached over plain OTLP.

A preset only knows: default endpoint, traces path, auth header shape and the
resource attributes that pick a project. Adding a vendor is a few lines here
or, with ``type: otlp``, pure configuration.

Third-party exporters (non-OTLP) can be added with
:func:`register_exporter` or the ``private_gpt.observability.exporters``
entry point group; the callable receives the exporter settings and returns an
OTel ``SpanExporter``.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from private_gpt.settings.settings import ObservabilityExporterSettings

logger = logging.getLogger(__name__)

ExporterFactory = Callable[["ObservabilityExporterSettings"], Any]

_CUSTOM_EXPORTERS: dict[str, ExporterFactory] = {}


def register_exporter(type_name: str, factory: ExporterFactory) -> None:
    """Register a custom exporter type usable as ``exporters[].type``."""
    _CUSTOM_EXPORTERS[type_name] = factory


def _basic(user: str, password: str) -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


def _preset(
    cfg: ObservabilityExporterSettings,
) -> tuple[str, dict[str, str], dict[str, str]]:
    """Return ``(endpoint, headers, resource_attributes)`` for ``cfg``."""
    opts = cfg.options
    headers: dict[str, str] = {}
    resource: dict[str, str] = {}
    endpoint = cfg.endpoint
    match cfg.type:
        case "phoenix":
            endpoint = endpoint or "http://localhost:6006"
            if cfg.api_key:
                headers["authorization"] = f"Bearer {cfg.api_key}"
            if cfg.project:
                resource["openinference.project.name"] = cfg.project
        case "arize":
            endpoint = endpoint or "https://otlp.arize.com"
            if cfg.api_key:
                headers["api_key"] = cfg.api_key
            if space_id := opts.get("space_id"):
                headers["space_id"] = space_id
            if cfg.project:
                resource["openinference.project.name"] = cfg.project
                resource["model_id"] = cfg.project
        case "langfuse":
            endpoint = (endpoint or "https://cloud.langfuse.com").rstrip("/")
            if (
                not endpoint.endswith("/api/public/otel")
                and "/v1/traces" not in endpoint
            ):
                endpoint += "/api/public/otel"
            public_key, secret_key = opts.get("public_key"), opts.get("secret_key")
            if public_key and secret_key:
                headers["authorization"] = _basic(public_key, secret_key)
            headers["x-langfuse-ingestion-version"] = "4"
        case "langsmith":
            endpoint = (endpoint or "https://api.smith.langchain.com").rstrip("/")
            if not endpoint.endswith("/otel") and "/v1/traces" not in endpoint:
                endpoint += "/otel"
            if cfg.api_key:
                headers["x-api-key"] = cfg.api_key
            if cfg.project:
                headers["Langsmith-Project"] = cfg.project
        case "opik":
            endpoint = (endpoint or "https://www.comet.com/opik/api").rstrip("/")
            if (
                not endpoint.endswith("/v1/private/otel")
                and "/v1/traces" not in endpoint
            ):
                endpoint += "/v1/private/otel"
            if cfg.api_key:
                headers["authorization"] = cfg.api_key
            if workspace := opts.get("workspace"):
                headers["Comet-Workspace"] = workspace
            if cfg.project:
                headers["projectName"] = cfg.project
        case "grafana":
            # Grafana Cloud OTLP gateway, or a local Alloy/Tempo when no
            # credentials are given.
            endpoint = endpoint or "http://localhost:4318"
            instance_id = opts.get("instance_id")
            if instance_id and cfg.api_key:
                headers["authorization"] = _basic(instance_id, cfg.api_key)
        case "datadog":
            site = opts.get("site", "datadoghq.com")
            endpoint = endpoint or f"https://otlp.{site}"
            if cfg.api_key:
                headers["dd-api-key"] = cfg.api_key
            headers["dd-otlp-source"] = "llmobs"
        case "honeycomb":
            endpoint = endpoint or "https://api.honeycomb.io"
            if cfg.api_key:
                headers["x-honeycomb-team"] = cfg.api_key
            if dataset := opts.get("dataset"):
                headers["x-honeycomb-dataset"] = dataset
        case _:
            endpoint = endpoint or "http://localhost:4318"
    headers.update(cfg.headers)
    return endpoint, headers, resource


# Backends built for LLM traces: by default they get only AI traces, not every
# HTTP request the API serves.
_LLM_BACKENDS = {"phoenix", "arize", "langfuse", "langsmith", "opik", "datadog"}


def exporter_tags(cfg: ObservabilityExporterSettings) -> list[str]:
    """Trace tags ``cfg`` receives (``all`` for every span)."""
    if cfg.tags is not None:
        return cfg.tags
    return ["ai"] if cfg.type in _LLM_BACKENDS else ["all"]


def _traces_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    return endpoint if endpoint.endswith("/v1/traces") else f"{endpoint}/v1/traces"


def build_exporter(cfg: ObservabilityExporterSettings) -> tuple[Any, dict[str, str]]:
    """Build the ``SpanExporter`` for one exporter entry."""
    if cfg.type in _CUSTOM_EXPORTERS:
        return _CUSTOM_EXPORTERS[cfg.type](cfg), {}
    for ep in entry_points(group="private_gpt.observability.exporters"):
        if ep.name == cfg.type:
            return ep.load()(cfg), {}

    if cfg.type == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        return ConsoleSpanExporter(), {}

    endpoint, headers, resource = _preset(cfg)
    if cfg.protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter as GrpcExporter,
        )

        return (
            GrpcExporter(
                endpoint=endpoint,
                headers=tuple((k.lower(), v) for k, v in headers.items()),
                insecure=endpoint.startswith("http://"),
            ),
            resource,
        )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter as HttpExporter,
    )

    return HttpExporter(endpoint=_traces_url(endpoint), headers=headers), resource


def build_span_processors(
    exporters: list[ObservabilityExporterSettings],
) -> tuple[list[Any], dict[str, str]]:
    """One ``BatchSpanProcessor`` per enabled exporter.

    Batch processing keeps exporting off the request path (the previous
    ``SimpleSpanProcessor`` did a blocking HTTP call per span) and the SDK
    restarts its worker thread after ``fork`` (Celery prefork).
    """
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from private_gpt.observability.processors import TagFilterProcessor

    processors: list[Any] = []
    resource: dict[str, str] = {}
    for cfg in exporters:
        if not cfg.enabled:
            continue
        try:
            exporter, exporter_resource = build_exporter(cfg)
        except Exception:
            logger.exception("Could not build %s trace exporter; skipping", cfg.type)
            continue
        processors.append(
            TagFilterProcessor(exporter_tags(cfg), BatchSpanProcessor(exporter))
        )
        for key, value in exporter_resource.items():
            # Resource is process-wide: first exporter that names a project wins.
            resource.setdefault(key, value)
    return processors, resource
