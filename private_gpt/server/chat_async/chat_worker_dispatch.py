from __future__ import annotations

from typing import TYPE_CHECKING, Any

from private_gpt.arq.enqueue import enqueue_start_chat_job

if TYPE_CHECKING:
    from private_gpt.components.chat.models.chat_config_models import ChatRequest
    from private_gpt.components.streaming.stream.stream_manager import StreamManager


async def create_stream_and_enqueue_chat(
    *,
    stream_manager: StreamManager,
    request: ChatRequest,
    message_id: str | None,
) -> str:
    metadata: dict[str, Any] = {
        "message_count": len(request.messages),
        "thinking_enabled": request.thinking.enabled,
    }
    correlation_id = await stream_manager.create_stream(
        stream_type="chat_completion",
        correlation_id=message_id,
        metadata=metadata,
    )
    await enqueue_start_chat_job(
        request_data=request.model_dump(mode="json"),
        correlation_id=correlation_id,
        stream_type="chat_completion",
        metadata=metadata,
        job_id=f"{correlation_id}:start",
    )
    return correlation_id
