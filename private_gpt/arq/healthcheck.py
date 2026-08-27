from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from arq.connections import create_pool
from arq.constants import in_progress_key_prefix
from arq.utils import timestamp_ms
from fastapi import FastAPI, HTTPException

from private_gpt.arq.liveness import read_worker_heartbeat
from private_gpt.arq.settings import get_healthcheck_redis_settings, get_queue_name
from private_gpt.settings.settings import settings as load_settings

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_STALE_SECONDS = 20
DEFAULT_IDLE_TIMEOUT_SECONDS = 10

app = FastAPI()


def _env_seconds(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


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


async def count_stale_pickable_jobs(
    redis: Any,
    *,
    queue_name: str,
    cutoff_ms: int,
) -> int:
    job_ids = await redis.zrangebyscore(
        queue_name,
        min=float("-inf"),
        max=cutoff_ms,
        start=0,
        num=50,
    )
    if not job_ids:
        return 0

    async with redis.pipeline(transaction=False) as pipe:
        for job_id in job_ids:
            decoded = job_id.decode() if isinstance(job_id, bytes) else job_id
            pipe.exists(in_progress_key_prefix + decoded)
        exists_flags = await pipe.execute()

    return sum(1 for exists in exists_flags if not exists)


async def probe_stale_pickable_jobs() -> tuple[int | None, str | None]:
    queue = os.environ.get("PGPT_ARQ_QUEUE", "").strip()
    if not queue:
        return None, "queue_not_configured"

    redis = None
    try:
        redis = await create_pool(get_healthcheck_redis_settings(load_settings()))
        idle_timeout_ms = (
            _env_seconds(
                "PGPT_ARQ_HEALTH_IDLE_TIMEOUT_SECONDS",
                DEFAULT_IDLE_TIMEOUT_SECONDS,
            )
            * 1000
        )
        count = await count_stale_pickable_jobs(
            redis,
            queue_name=get_queue_name(queue),
            cutoff_ms=timestamp_ms() - idle_timeout_ms,
        )
        return count, None
    except Exception as exc:
        logger.critical("Error checking ARQ queue health: %s", exc)
        return None, "redis_unreachable"
    finally:
        if redis is not None:
            try:
                await redis.aclose()
            except Exception:
                logger.debug(
                    "Failed to close ARQ healthcheck redis pool", exc_info=True
                )


async def check_health() -> dict[str, Any]:
    try:
        heartbeat = read_worker_heartbeat()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return _unhealthy("invalid_heartbeat")

    if heartbeat is None:
        return _unhealthy("not_ready")

    age_seconds = time.time() - heartbeat.ts
    stale_after = _env_seconds(
        "PGPT_ARQ_HEALTH_HEARTBEAT_STALE_SECONDS",
        DEFAULT_HEARTBEAT_STALE_SECONDS,
    )
    heartbeat_stale = age_seconds > stale_after

    if heartbeat.ongoing > 0:
        return _healthy(ongoing=heartbeat.ongoing, max_jobs=heartbeat.max_jobs)

    if heartbeat_stale:
        return _unhealthy(
            "heartbeat_stale",
            ongoing=heartbeat.ongoing,
            heartbeat_age_seconds=round(age_seconds, 2),
        )

    pending, error = await probe_stale_pickable_jobs()
    if error is not None:
        return _unhealthy(error, ongoing=heartbeat.ongoing)
    if pending:
        return _unhealthy(
            "idle_with_pending_jobs",
            ongoing=heartbeat.ongoing,
            pending_jobs=pending,
        )

    return _healthy(ongoing=heartbeat.ongoing, max_jobs=heartbeat.max_jobs)


@app.get("/health")
async def health() -> dict[str, Any]:
    status = await check_health()
    if status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=status)
    return status
