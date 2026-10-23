from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from arq.connections import create_pool
from arq.utils import timestamp_ms
from fastapi import FastAPI, HTTPException

from private_gpt.arq.leases import (
    decode_in_progress_owner,
    env_seconds,
    in_progress_key,
    worker_lease_key,
)
from private_gpt.arq.liveness import arq_health_check_key, worker_is_ready
from private_gpt.arq.routing import (
    decode_route,
    expected_queue_for_worker,
    route_key,
)
from private_gpt.arq.settings import (
    get_control_redis_settings,
    get_healthcheck_redis_settings,
    get_queue_name,
    get_redis_settings,
)
from private_gpt.settings.settings import settings as load_settings

if TYPE_CHECKING:
    from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_IDLE_TIMEOUT_SECONDS = 2
_ARQ_HEALTH_PATTERN = re.compile(
    r"\bj_complete=(?P<complete>\d+)\s+"
    r"j_failed=(?P<failed>\d+)\s+"
    r"j_retried=(?P<retried>\d+)\s+"
    r"j_ongoing=(?P<ongoing>\d+)\s+"
    r"queued=(?P<queued>\d+)\b"
)

app = FastAPI()


@dataclass(frozen=True)
class ArqHealthState:
    complete: int
    failed: int
    retried: int
    ongoing: int
    queued: int


@dataclass(frozen=True)
class DueJobCounts:
    pickable: int = 0
    stale_in_progress: int = 0
    unowned_in_progress: int = 0


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


def _worker_type() -> str | None:
    worker_type = os.environ.get("PGPT_STATEFUL_WORKER_TYPE", "").strip()
    return worker_type or None


def _configured_queue() -> str | None:
    queue = os.environ.get("PGPT_ARQ_QUEUE", "").strip()
    return get_queue_name(queue) if queue else None


def validate_worker_route_settings(settings: Settings) -> dict[str, Any] | None:
    worker_type = _worker_type()
    actual_queue = _configured_queue()
    if actual_queue is None:
        return _unhealthy("queue_not_configured")
    if worker_type is None:
        return None

    expected_queue = expected_queue_for_worker(settings, worker_type)
    if expected_queue is not None and actual_queue != expected_queue:
        return _unhealthy(
            "queue_config_mismatch",
            worker_type=worker_type,
            expected_queue=expected_queue,
            actual_queue=actual_queue,
        )
    return None


def parse_arq_health(value: bytes | str) -> ArqHealthState:
    decoded = value.decode() if isinstance(value, bytes) else value
    match = _ARQ_HEALTH_PATTERN.search(decoded)
    if match is None:
        raise ValueError(f"invalid ARQ health value: {decoded!r}")
    fields = {name: int(raw) for name, raw in match.groupdict().items()}
    return ArqHealthState(**fields)


async def inspect_due_jobs(
    redis: Any,
    *,
    queue_name: str,
    cutoff_ms: int,
) -> DueJobCounts:
    job_ids = await redis.zrangebyscore(
        queue_name,
        min=float("-inf"),
        max=cutoff_ms,
        start=0,
        num=50,
    )
    if not job_ids:
        return DueJobCounts()

    decoded_ids = [
        job_id.decode() if isinstance(job_id, bytes) else str(job_id)
        for job_id in job_ids
    ]
    async with redis.pipeline(transaction=False) as pipe:
        for job_id in decoded_ids:
            pipe.get(in_progress_key(job_id))
        lock_values = await pipe.execute()

    pickable = 0
    unowned = 0
    owned: list[str] = []
    for lock_value in lock_values:
        if lock_value is None:
            pickable += 1
            continue
        owner = decode_in_progress_owner(lock_value)
        if owner is None:
            unowned += 1
        else:
            owned.append(owner)

    stale = 0
    if owned:
        async with redis.pipeline(transaction=False) as pipe:
            for owner in owned:
                pipe.exists(worker_lease_key(owner))
            lease_exists = await pipe.execute()
        stale = sum(1 for exists in lease_exists if not exists)

    return DueJobCounts(
        pickable=pickable,
        stale_in_progress=stale,
        unowned_in_progress=unowned,
    )


async def count_stale_pickable_jobs(
    redis: Any,
    *,
    queue_name: str,
    cutoff_ms: int,
) -> int:
    counts = await inspect_due_jobs(
        redis,
        queue_name=queue_name,
        cutoff_ms=cutoff_ms,
    )
    return counts.pickable


async def probe_worker_state(
    settings: Settings,
) -> tuple[ArqHealthState | None, DueJobCounts | None, str | None]:
    queue_name = _configured_queue()
    if queue_name is None:
        return None, None, "queue_not_configured"

    redis = None
    try:
        redis = await create_pool(get_healthcheck_redis_settings(settings))
        raw_health = await redis.get(arq_health_check_key(queue_name))
        if raw_health is None:
            return None, None, "arq_health_missing"
        try:
            arq_health = parse_arq_health(raw_health)
        except ValueError:
            return None, None, "invalid_arq_health"

        idle_timeout_ms = (
            env_seconds(
                "PGPT_ARQ_HEALTH_IDLE_TIMEOUT_SECONDS",
                DEFAULT_IDLE_TIMEOUT_SECONDS,
            )
            * 1000
        )
        due_jobs = await inspect_due_jobs(
            redis,
            queue_name=queue_name,
            cutoff_ms=timestamp_ms() - idle_timeout_ms,
        )
        return arq_health, due_jobs, None
    except Exception as exc:
        logger.critical("Error checking ARQ worker health: %s", exc)
        return None, None, "redis_unreachable"
    finally:
        if redis is not None:
            try:
                await redis.aclose()
            except Exception:
                logger.debug(
                    "Failed to close ARQ healthcheck redis pool", exc_info=True
                )


async def probe_publisher_route(settings: Settings) -> dict[str, Any] | None:
    worker_type = _worker_type()
    actual_queue = _configured_queue()
    if worker_type is None or actual_queue is None:
        return None

    redis = None
    try:
        redis = await create_pool(get_control_redis_settings(settings))
        raw_route = await redis.get(route_key(worker_type))
    except Exception as exc:
        logger.critical("Error checking ARQ publisher route: %s", exc)
        return _unhealthy("route_registry_unreachable")
    finally:
        if redis is not None:
            try:
                await redis.aclose()
            except Exception:
                logger.debug("Failed to close ARQ route pool", exc_info=True)

    if raw_route is None:
        return None
    try:
        route = decode_route(raw_route)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return _unhealthy("invalid_publisher_route")

    actual_database = get_redis_settings(settings).database
    if route.queue_name != actual_queue or route.database != actual_database:
        return _unhealthy(
            "publisher_route_mismatch",
            worker_type=worker_type,
            publisher_queue=route.queue_name,
            worker_queue=actual_queue,
            publisher_database=route.database,
            worker_database=actual_database,
        )
    return None


async def check_health() -> dict[str, Any]:
    current_settings = load_settings()
    route_settings_error = validate_worker_route_settings(current_settings)
    if route_settings_error is not None:
        return route_settings_error

    if not worker_is_ready():
        return _unhealthy("not_ready")

    arq_health, due_jobs, error = await probe_worker_state(current_settings)
    if error is not None:
        return _unhealthy(error)
    assert arq_health is not None
    assert due_jobs is not None

    publisher_route_error = await probe_publisher_route(current_settings)
    if publisher_route_error is not None:
        return publisher_route_error

    if due_jobs.stale_in_progress:
        return _unhealthy(
            "stale_in_progress_jobs",
            ongoing=arq_health.ongoing,
            stale_in_progress_jobs=due_jobs.stale_in_progress,
        )
    if due_jobs.unowned_in_progress:
        return _unhealthy(
            "unowned_in_progress_jobs",
            ongoing=arq_health.ongoing,
            unowned_in_progress_jobs=due_jobs.unowned_in_progress,
        )
    if due_jobs.pickable and arq_health.ongoing == 0:
        return _unhealthy(
            "idle_with_pending_jobs",
            ongoing=arq_health.ongoing,
            pending_jobs=due_jobs.pickable,
        )

    return _healthy(
        ongoing=arq_health.ongoing,
        queued=arq_health.queued,
        max_jobs=int(os.environ.get("PGPT_ARQ_MAX_JOBS", "0")),
    )


@app.get("/health")
async def health() -> dict[str, Any]:
    status = await check_health()
    if status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=status)
    return status
