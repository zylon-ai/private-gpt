"""Process-wide OpenTelemetry setup: one provider, N exporters, M instrumentors.

Called once per process (API, Celery worker, ARQ worker) from
``private_gpt.initialize.initialize_observability``. Nothing here is
vendor-specific beyond the small preset table in
:mod:`private_gpt.observability.exporters`.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING, Any

from private_gpt.utils.dependencies import format_missing_dependency_message

if TYPE_CHECKING:
    from private_gpt.settings.settings import ObservabilitySettings, Settings

logger = logging.getLogger(__name__)

_configured = False
_capture_content = True
_settings: ObservabilitySettings | None = None
_provider: Any = None


def is_configured() -> bool:
    return _configured


def capture_content() -> bool:
    return _capture_content


def observability_settings() -> ObservabilitySettings | None:
    return _settings


def process_role() -> str:
    """Best-effort name of the current process kind for ``service.name``."""
    if role := os.environ.get("PGPT_PROCESS_ROLE"):
        return role
    argv = " ".join(sys.argv).lower()
    if "celery" in argv:
        return "celery"
    if "arq" in argv or "worker" in argv:
        return "worker"
    return "api"


def configure_tracing(settings: Settings, *, role: str | None = None) -> bool:
    """Install the global tracer provider and library instrumentations.

    Idempotent: subsequent calls in the same process are ignored, which makes
    it safe for every entrypoint (API factory, Celery import, ARQ startup) to
    call it.
    """
    global _configured, _capture_content, _settings, _provider
    if _configured:
        return True

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message(
                "OpenTelemetry tracing", extras="observability-otel"
            )
        ) from e

    from private_gpt.observability.exporters import build_span_processors
    from private_gpt.observability.instrumentations import instrument_libraries
    from private_gpt.observability.processors import (
        BackgroundSpanProcessor,
        DropSpansProcessor,
        FanOutSpanProcessor,
        SemconvSpanProcessor,
        TraceTagsProcessor,
    )
    from private_gpt.utils.runner import get_version

    obs = settings.observability
    _settings = obs
    _capture_content = obs.capture_content
    role = role or process_role()

    resource_attributes: dict[str, Any] = {
        "service.name": obs.service_name or f"{obs.service_namespace}-{role}",
        "service.namespace": obs.service_namespace,
        "service.version": get_version(),
        "pgpt.process.role": role,
        **obs.resource_attributes,
    }
    if obs.environment:
        resource_attributes["deployment.environment.name"] = obs.environment

    exporter_processors = build_span_processors(obs.exporters)
    if not exporter_processors:
        logger.warning(
            "Tracing is on but no exporter is configured; spans are kept "
            "in-process only (useful for propagation and log correlation)."
        )

    provider = TracerProvider(
        resource=Resource.create(resource_attributes),
        sampler=ParentBased(TraceIdRatioBased(obs.sampling_ratio)),
    )
    provider.add_span_processor(
        # Tagging sees every span (dropped ones too) so per-trace bookkeeping
        # always balances; exporters then filter on the tags. All of it runs
        # in a background thread: a request only enqueues finished spans.
        BackgroundSpanProcessor(  # ty: ignore[invalid-argument-type] - duck-typed SpanProcessor
            TraceTagsProcessor(
                rules=obs.trace_tags,
                delegate=DropSpansProcessor(
                    patterns=obs.drop_spans,
                    delegate=SemconvSpanProcessor(
                        semconv=obs.semconv,
                        delegate=FanOutSpanProcessor(exporter_processors),
                    ),
                ),
            )
        )
    )
    trace.set_tracer_provider(provider)
    _provider = provider
    _configured = True

    instrument_libraries(obs, provider)
    logger.info(
        "Tracing enabled: service=%s exporters=%s semconv=%s",
        resource_attributes["service.name"],
        [e.type for e in obs.exporters if e.enabled],
        obs.semconv,
    )
    return True


def shutdown_tracing() -> None:
    """Flush pending spans; registered on process exit by the SDK as well."""
    if _provider is not None:
        try:
            _provider.force_flush(timeout_millis=5_000)
            _provider.shutdown()
        except Exception:  # pragma: no cover - best effort on exit
            logger.debug("Error flushing spans on shutdown", exc_info=True)


def _reset_for_tests() -> None:
    global _configured, _settings, _provider, _capture_content
    _configured = False
    _settings = None
    _provider = None
    _capture_content = True
