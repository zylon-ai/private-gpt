from typing import Any

from arq import create_pool

from private_gpt.arq.settings import CHAT_TASK_NAME, get_queue_name, get_redis_settings
from private_gpt.settings.settings import settings as _settings


async def enqueue_chat_job(
    *,
    body: dict[str, Any],
    correlation_id: str,
    stream_type: str,
    metadata: dict[str, Any],
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
            _job_id=correlation_id,
            _queue_name=get_queue_name(current_settings),
        )
    finally:
        await redis.aclose()
