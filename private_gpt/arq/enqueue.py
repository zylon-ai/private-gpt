import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

from arq import create_pool
from arq.jobs import Job

from private_gpt.arq.settings import get_redis_settings
from private_gpt.settings.settings import settings as _settings

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task[Any]] = set()


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
    redis = await create_pool(get_redis_settings(current_settings))
    try:
        options: dict[str, Any] = {"_queue_name": queue_name}
        if job_id is not None:
            options["_job_id"] = job_id
        if defer_seconds is not None:
            options["_defer_by"] = defer_seconds
        return await redis.enqueue_job(task_name, *args, **options) is not None
    finally:
        await redis.aclose()


def _spawn(coro: Coroutine[Any, Any, Any], *, name: str) -> None:
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _abort_job(*, job_id: str, queue_name: str, timeout: int) -> bool:
    current_settings = _settings()
    redis = await create_pool(get_redis_settings(current_settings))
    try:
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
    finally:
        await redis.aclose()


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
