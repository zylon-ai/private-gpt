import logging
from typing import Any

from arq import create_pool
from arq.jobs import Job

from private_gpt.arq.settings import get_queue_name, get_redis_settings
from private_gpt.settings.settings import settings as _settings

logger = logging.getLogger(__name__)


def _log_dispatch(
    *,
    task_name: str,
    queue_name: str,
    job_id: str | None,
    correlation_id: str,
    defer_seconds: int | None = None,
) -> None:
    logger.info(
        "Dispatching ARQ task=%s queue=%s job_id=%s correlation_id=%s "
        "defer_seconds=%s",
        task_name,
        queue_name,
        job_id,
        correlation_id,
        defer_seconds,
    )


async def enqueue_job(
    *,
    task_name: str,
    args: tuple[Any, ...] = (),
    correlation_id: str,
    job_id: str | None = None,
    defer_seconds: int | None = None,
) -> None:
    current_settings = _settings()
    queue_name = get_queue_name(current_settings)
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
        await redis.enqueue_job(task_name, *args, **options)
    finally:
        await redis.aclose()


async def abort_job(*, job_id: str) -> bool:
    current_settings = _settings()
    redis = await create_pool(get_redis_settings(current_settings))
    try:
        job = Job(
            job_id,
            redis=redis,
            _queue_name=get_queue_name(current_settings),
        )
        return await job.abort(timeout=2)
    finally:
        await redis.aclose()
