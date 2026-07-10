from typing import Any

from arq import create_pool
from arq.jobs import Job

from private_gpt.arq.settings import (
    RESUME_ITERATION_TASK_NAME,
    START_CHAT_TASK_NAME,
    TOOL_RESUME_TASK_NAME,
    get_queue_name,
    get_redis_settings,
)
from private_gpt.settings.settings import settings as _settings


async def enqueue_start_chat_job(
    *,
    body: dict[str, Any],
    correlation_id: str,
    stream_type: str,
    metadata: dict[str, Any],
    job_id: str | None = None,
) -> None:
    current_settings = _settings()
    redis = await create_pool(get_redis_settings(current_settings))
    try:
        await redis.enqueue_job(
            START_CHAT_TASK_NAME,
            body,
            correlation_id,
            stream_type,
            metadata,
            _job_id=job_id or f"{correlation_id}:start",
            _keep_result=10,
            _queue_name=get_queue_name(current_settings),
        )
    finally:
        await redis.aclose()


async def enqueue_resume_iteration_job(
    *,
    body: dict[str, Any],
    correlation_id: str,
    stream_type: str,
    metadata: dict[str, Any],
    request_data: dict[str, Any],
    pause_type: str,
    tool_results: list[dict[str, Any]] | None = None,
    iteration: int,
    next_block_count: int,
    job_id: str | None = None,
) -> None:
    current_settings = _settings()
    redis = await create_pool(get_redis_settings(current_settings))
    try:
        await redis.enqueue_job(
            RESUME_ITERATION_TASK_NAME,
            body,
            correlation_id,
            stream_type,
            metadata,
            request_data,
            pause_type,
            tool_results or [],
            iteration,
            next_block_count,
            _job_id=job_id or f"{correlation_id}:resume:{iteration}",
            _keep_result=10,
            _queue_name=get_queue_name(current_settings),
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
    redis = await create_pool(get_redis_settings(current_settings))
    try:
        await redis.enqueue_job(
            TOOL_RESUME_TASK_NAME,
            correlation_id,
            tool_id,
            result,
            _keep_result=10,
            _queue_name=get_queue_name(current_settings),
        )
    finally:
        await redis.aclose()


async def abort_chat_job(*, correlation_id: str) -> bool:
    current_settings = _settings()
    redis = await create_pool(get_redis_settings(current_settings))
    try:
        job = Job(
            f"{correlation_id}:start",
            redis=redis,
            _queue_name=get_queue_name(current_settings),
        )
        return await job.abort(timeout=2)
    finally:
        await redis.aclose()
