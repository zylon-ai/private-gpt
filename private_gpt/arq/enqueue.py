import asyncio
import logging
from typing import Any

from arq import create_pool
from arq.jobs import Job

from private_gpt.arq.settings import (
    CHAT_TIMEOUT_TASK_NAME,
    RESUME_ITERATION_TASK_NAME,
    START_CHAT_TASK_NAME,
    TOOL_RESUME_TASK_NAME,
    get_queue_name,
    get_redis_settings,
)
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


async def enqueue_start_chat_job(
    *,
    request_data: dict[str, Any],
    correlation_id: str,
    stream_type: str,
    metadata: dict[str, Any],
    job_id: str | None = None,
) -> None:
    current_settings = _settings()
    queue_name = get_queue_name(current_settings)
    resolved_job_id = job_id or f"{correlation_id}:start"
    _log_dispatch(
        task_name=START_CHAT_TASK_NAME,
        queue_name=queue_name,
        job_id=resolved_job_id,
        correlation_id=correlation_id,
    )
    redis = await create_pool(get_redis_settings(current_settings))
    try:
        await redis.enqueue_job(
            START_CHAT_TASK_NAME,
            request_data,
            correlation_id,
            stream_type,
            metadata,
            _job_id=resolved_job_id,
            _queue_name=queue_name,
        )
    finally:
        await redis.aclose()


async def enqueue_resume_iteration_job(
    *,
    correlation_id: str,
    job_id: str | None = None,
) -> None:
    current_settings = _settings()
    queue_name = get_queue_name(current_settings)
    resolved_job_id = job_id or f"{correlation_id}:resume"
    _log_dispatch(
        task_name=RESUME_ITERATION_TASK_NAME,
        queue_name=queue_name,
        job_id=resolved_job_id,
        correlation_id=correlation_id,
    )
    redis = await create_pool(get_redis_settings(current_settings))
    try:
        await redis.enqueue_job(
            RESUME_ITERATION_TASK_NAME,
            correlation_id,
            _job_id=resolved_job_id,
            _queue_name=queue_name,
        )
    finally:
        await redis.aclose()


async def enqueue_chat_timeout_job(
    *,
    correlation_id: str,
    checkpoint_id: str,
    delay_seconds: int,
    job_id: str,
) -> None:
    current_settings = _settings()
    queue_name = get_queue_name(current_settings)
    _log_dispatch(
        task_name=CHAT_TIMEOUT_TASK_NAME,
        queue_name=queue_name,
        job_id=job_id,
        correlation_id=correlation_id,
        defer_seconds=delay_seconds,
    )
    redis = await create_pool(get_redis_settings(current_settings))
    try:
        await redis.enqueue_job(
            CHAT_TIMEOUT_TASK_NAME,
            correlation_id,
            checkpoint_id,
            _defer_by=delay_seconds,
            _job_id=job_id,
            _queue_name=queue_name,
        )
    finally:
        await redis.aclose()


async def enqueue_tool_resume_job(
    *,
    correlation_id: str,
    tool_id: str,
    result: dict[str, Any],
) -> None:
    current_settings = _settings()
    queue_name = get_queue_name(current_settings)
    _log_dispatch(
        task_name=TOOL_RESUME_TASK_NAME,
        queue_name=queue_name,
        job_id=None,
        correlation_id=correlation_id,
    )
    redis = await create_pool(get_redis_settings(current_settings))
    try:
        await redis.enqueue_job(
            TOOL_RESUME_TASK_NAME,
            correlation_id,
            tool_id,
            result,
            _queue_name=queue_name,
        )
    finally:
        await redis.aclose()


async def abort_chat_job(*, correlation_id: str) -> bool:
    results = await asyncio.gather(
        abort_job(job_id=f"{correlation_id}:start"),
        abort_job(job_id=f"{correlation_id}:resume"),
    )
    return any(results)


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
