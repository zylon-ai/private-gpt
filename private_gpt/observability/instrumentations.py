"""Explicit registry of library instrumentations (no auto-discovery).

``opentelemetry-distro`` style auto-loading of every installed instrumentor
conflicts with Celery prefork and per-app FastAPI options, so each one is
listed here with its kwargs and enabled by name from settings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from private_gpt.settings.settings import ObservabilitySettings

logger = logging.getLogger(__name__)

LEAN = ("httpx", "threading", "celery", "logging", "llama_index", "mcp", "fastapi")
FULL = (*LEAN, "grpc", "sqlalchemy", "redis")

_enabled: set[str] = set()


def _httpx(_: ObservabilitySettings, provider: Any) -> None:
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    HTTPXClientInstrumentor().instrument(tracer_provider=provider)


def _threading(_: ObservabilitySettings, provider: Any) -> None:
    # Propagates the current context through Thread.start and
    # ThreadPoolExecutor.submit (and therefore loop.run_in_executor).
    from opentelemetry.instrumentation.threading import ThreadingInstrumentor

    ThreadingInstrumentor().instrument(tracer_provider=provider)


def _celery(_: ObservabilitySettings, provider: Any) -> None:
    from opentelemetry.instrumentation.celery import CeleryInstrumentor

    CeleryInstrumentor().instrument(tracer_provider=provider)


def _logging(_: ObservabilitySettings, provider: Any) -> None:
    # Adds otelTraceID / otelSpanID / otelServiceName to every LogRecord so log
    # lines can be correlated with traces (format is left untouched).
    from opentelemetry.instrumentation.logging import LoggingInstrumentor

    LoggingInstrumentor().instrument(tracer_provider=provider, set_logging_format=False)


def _llama_index(obs: ObservabilitySettings, provider: Any) -> None:
    from openinference.instrumentation import TraceConfig
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    from private_gpt.observability.llama_index_patch import patch_llama_index_handler

    patch_llama_index_handler(obs.max_llm_input_messages)
    hide = not obs.capture_content
    LlamaIndexInstrumentor().instrument(
        tracer_provider=provider,
        use_legacy_callback_handler=False,
        config=TraceConfig(
            hide_inputs=hide,
            hide_outputs=hide,
            hide_embedding_vectors=True,
            base64_image_max_length=8_000,
        ),
    )
    logging.getLogger("openinference.instrumentation.llama_index._handler").setLevel(
        logging.CRITICAL
    )
    logging.getLogger("opentelemetry.attributes").setLevel(logging.CRITICAL)


def _mcp(_: ObservabilitySettings, provider: Any) -> None:
    # mcp>=2 propagates traceparent in request `_meta` and emits its own spans
    # whenever a global tracer provider is set; nothing to patch here.
    del provider


def _fastapi(_: ObservabilitySettings, provider: Any) -> None:
    # FastAPI is instrumented per app in ``instrument_fastapi_app`` because the
    # app is built by a factory; enabling the name here only flags it.
    del provider


def _grpc(_: ObservabilitySettings, provider: Any) -> None:
    # Patches grpc channel factories: tritonclient channels created afterwards
    # get client spans and forward traceparent in gRPC metadata.
    from opentelemetry.instrumentation.grpc import (
        GrpcAioInstrumentorClient,
        GrpcInstrumentorClient,
    )

    GrpcInstrumentorClient().instrument(tracer_provider=provider)
    GrpcAioInstrumentorClient().instrument(tracer_provider=provider)


def _sqlalchemy(_: ObservabilitySettings, provider: Any) -> None:
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    SQLAlchemyInstrumentor().instrument(tracer_provider=provider)


def _redis(_: ObservabilitySettings, provider: Any) -> None:
    from opentelemetry.instrumentation.redis import RedisInstrumentor

    RedisInstrumentor().instrument(tracer_provider=provider)


_INSTRUMENTATIONS: dict[str, Callable[[ObservabilitySettings, Any], None]] = {
    "httpx": _httpx,
    "threading": _threading,
    "celery": _celery,
    "logging": _logging,
    "llama_index": _llama_index,
    "mcp": _mcp,
    "fastapi": _fastapi,
    "grpc": _grpc,
    "sqlalchemy": _sqlalchemy,
    "redis": _redis,
}


def register_instrumentation(
    name: str, installer: Callable[[ObservabilitySettings, Any], None]
) -> None:
    """Let extensions add named instrumentations."""
    _INSTRUMENTATIONS[name] = installer


def resolve(obs: ObservabilitySettings) -> list[str]:
    match obs.instrumentations:
        case "none":
            return []
        case "lean":
            return list(LEAN)
        case "full":
            return list(FULL)
        case names:
            return list(names)


def is_enabled(name: str) -> bool:
    return name in _enabled


def instrument_libraries(obs: ObservabilitySettings, provider: Any) -> None:
    for name in resolve(obs):
        installer = _INSTRUMENTATIONS.get(name)
        if installer is None:
            logger.warning("Unknown instrumentation %r; skipping", name)
            continue
        try:
            installer(obs, provider)
        except ImportError as e:
            logger.warning("Instrumentation %r unavailable: %s", name, e)
            continue
        except Exception:
            logger.exception("Instrumentation %r failed", name)
            continue
        _enabled.add(name)


def instrument_fastapi_app(app: Any) -> None:
    """Instrument one FastAPI app (called from the app factory)."""
    from private_gpt.observability import setup

    obs = setup.observability_settings()
    if obs is None or not is_enabled("fastapi"):
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError as e:
        logger.warning("FastAPI instrumentation unavailable: %s", e)
        return
    from private_gpt.observability.http import HttpBodyCaptureMiddleware

    # Added before instrumenting so it runs inside the server span.
    app.add_middleware(HttpBodyCaptureMiddleware)
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls=",".join(obs.excluded_urls),
        # One span per SSE chunk is pure noise; the server span already covers
        # the full streaming duration.
        exclude_spans=["send", "receive"],
        server_request_hook=_server_request_hook,
    )


def _server_request_hook(span: Any, scope: dict[str, Any]) -> None:
    if not span or not span.is_recording():
        return
    headers = dict(scope.get("headers") or [])
    for header, attribute in (
        (b"x-request-id", "http.request.id"),
        (b"anthropic-beta", "pgpt.anthropic_beta"),
        (b"user-agent", "user_agent.original"),
    ):
        if value := headers.get(header):
            span.set_attribute(attribute, value.decode(errors="replace")[:256])
