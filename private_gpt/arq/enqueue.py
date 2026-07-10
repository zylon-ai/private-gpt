from typing import Any

from arq import create_pool
from arq.jobs import Job

from private_gpt.arq.settings import (
    CHAT_TASK_NAME,
    TOOL_DONE_TASK_NAME,
    get_queue_name,
    get_redis_settings,
)
from private_gpt.settings.settings import settings as _settings


async def enqueue_chat_job(
    *,
    body: dict[str, Any],
    correlation_id: str,
    stream_type: str,
    metadata: dict[str, Any],
    request_data: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> None:
    current_settings = _settings()
    redis = await create_pool(get_redis_settings(current_settings))
    try:
        await redis.enqueue_job(
            CHAT_TASK_NAME,
            body,
            correlation_id,
            stream_type,
            metadata,
            request_data,
            _job_id=job_id or correlation_id,
            _keep_result=10,
            _queue_name=get_queue_name(current_settings),
        )
    finally:
        await redis.aclose()


async def enqueue_tool_done_job(
    *,
    correlation_id: str,
    tool_id: str,
    result: dict[str, Any],
) -> None:
    current_settings = _settings()
    redis = await create_pool(get_redis_settings(current_settings))
    try:
        await redis.enqueue_job(
            TOOL_DONE_TASK_NAME,
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
            correlation_id,
            redis=redis,
            _queue_name=get_queue_name(current_settings),
        )
        return await job.abort(timeout=2)
    finally:
        await redis.aclose()
