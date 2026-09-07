from __future__ import annotations

import logging
import os
from typing import Any

from arq.connections import create_pool
from fastapi import FastAPI, HTTPException

from private_gpt.arq.settings import (
    arq_health_check_key,
    get_healthcheck_redis_settings,
    get_queue_name,
)
from private_gpt.settings.settings import settings as load_settings

logger = logging.getLogger(__name__)

app = FastAPI()


def _unhealthy(reason: str, **extra: Any) -> dict[str, Any]:
    status: dict[str, Any] = {
        "status": "unhealthy",
        "mode": "arq-worker",
        "services": {"worker": "unhealthy"},
        "reason": reason,
    }
    status.update(extra)
    return status


def _healthy(**extra: Any) -> dict[str, Any]:
    status: dict[str, Any] = {
        "status": "healthy",
        "mode": "arq-worker",
        "services": {"worker": "healthy"},
    }
    status.update(extra)
    return status


def _configured_queue() -> str | None:
    queue = os.environ.get("PGPT_ARQ_QUEUE", "").strip()
    return get_queue_name(queue) if queue else None


async def check_health() -> dict[str, Any]:
    """Check liveness the same way ARQ does: the worker's Redis sentinel exists.

    ARQ writes ``{queue}:health-check`` (we namespace it per pod) every
    ``health_check_interval`` seconds with a TTL of interval+1. ``arq --check``
    is healthy iff that key is present. We do the same GET and nothing else.
    """
    queue_name = _configured_queue()
    if queue_name is None:
        return _unhealthy("queue_not_configured")

    health_key = arq_health_check_key(queue_name)
    try:
        redis = await create_pool(get_healthcheck_redis_settings(load_settings()))
        try:
            data = await redis.get(health_key)
        finally:
            await redis.aclose()
    except Exception as exc:
        logger.critical("Error checking ARQ worker health: %s", exc)
        return _unhealthy("redis_unreachable")

    if not data:
        return _unhealthy("arq_health_missing")
    return _healthy()


@app.get("/health")
async def health() -> dict[str, Any]:
    status = await check_health()
    if status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=status)
    return status
