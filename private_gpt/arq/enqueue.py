import asyncio
import logging
import time
from collections.abc import Coroutine
from typing import Any

from arq import ArqRedis, create_pool
from arq.jobs import Job

from private_gpt.arq.routing import publish_route
from private_gpt.arq.settings import get_redis_settings
from private_gpt.settings.settings import settings as _settings

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task[Any]] = set()

# One pool per event loop: a pool per enqueue/abort opened a new connection
# (HELLO/CLIENT SETINFO, SELECT, PING) on every chat request.
# Keyed by loop: Celery tasks run short-lived loops, so pools of closed loops
# are dropped (their connections died with the loop).
_pools: dict[asyncio.AbstractEventLoop, ArqRedis] = {}
# The publisher route has a 24 h TTL; refreshing it once a minute is plenty and
# saves a second pool and a SETEX per request.
_ROUTE_REFRESH_SECONDS = 60.0
_route_published_at: dict[tuple[str, str], float] = {}


async def _get_pool() -> ArqRedis:
    loop = asyncio.get_running_loop()
    pool = _pools.get(loop)
    if pool is None:
        for stale in [other for other in _pools if other.is_closed()]:
            del _pools[stale]
        pool = await create_pool(get_redis_settings(_settings()))
        _pools[loop] = pool
    return pool


async def _publish_route_throttled(
    current_settings: Any, *, worker_type: str, queue_name: str
) -> None:
    key = (worker_type, queue_name)
    now = time.monotonic()
    published_at = _route_published_at.get(key)
    if published_at is not None and now - published_at < _ROUTE_REFRESH_SECONDS:
        return
    await publish_route(
        current_settings, worker_type=worker_type, queue_name=queue_name
    )
    _route_published_at[key] = now


def _log_dispatch(
    *,
    task_name: str,
    queue_name: str,
    job_id: str | None,
    correlation_id: str,
    defer_seconds: int | None = None,
) -> None:
    logger.info(
        "Dispatching ARQ task=%s queue=%s job_id=%s correlation_id=%s defer_seconds=%s",
        task_name,
        queue_name,
        job_id,
        correlation_id,
        defer_seconds,
    )


async def enqueue_job(
    *,
    task_name: str,
    queue_name: str,
    args: tuple[Any, ...] = (),
    correlation_id: str,
    worker_type: str,
    job_id: str | None = None,
    defer_seconds: int | None = None,
) -> bool:
    current_settings = _settings()
    _log_dispatch(
        task_name=task_name,
        queue_name=queue_name,
        job_id=job_id,
        correlation_id=correlation_id,
        defer_seconds=defer_seconds,
    )
    try:
        await _publish_route_throttled(
            current_settings,
            worker_type=worker_type,
            queue_name=queue_name,
        )
    except Exception:
        logger.exception(
            "Failed to publish ARQ route worker_type=%s queue=%s",
            worker_type,
            queue_name,
        )
    options: dict[str, Any] = {"_queue_name": queue_name}
    if job_id is not None:
        options["_job_id"] = job_id
    if defer_seconds is not None:
        options["_defer_by"] = defer_seconds
    redis = await _get_pool()
    return await redis.enqueue_job(task_name, *args, **options) is not None


def _spawn(coro: Coroutine[Any, Any, Any], *, name: str) -> None:
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _abort_job(*, job_id: str, queue_name: str, timeout: int) -> bool:
    redis = await _get_pool()
    job = Job(
        job_id,
        redis=redis,
        _queue_name=queue_name,
    )
    try:
        return await job.abort(timeout=timeout)
    except TimeoutError:
        logger.warning(
            "Timed out confirming abort for job_id=%s queue=%s",
            job_id,
            queue_name,
        )
        return False


async def abort_job(
    *,
    job_id: str,
    queue_name: str,
    wait: bool = False,
    timeout: int = 5,
) -> bool:
    if wait:
        return await _abort_job(
            job_id=job_id,
            queue_name=queue_name,
            timeout=timeout,
        )

    _spawn(
        _abort_job(
            job_id=job_id,
            queue_name=queue_name,
            timeout=timeout,
        ),
        name=f"abort_job_{job_id}",
    )
    return True
