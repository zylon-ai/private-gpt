from __future__ import annotations

from typing import TYPE_CHECKING

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.tools.remote_execution import ToolExecutionResponse

if TYPE_CHECKING:
    from private_gpt.arq.iteration_state import IterationStateService
    from private_gpt.components.engines.chat.async_chat_engine import (
        EventChannel,
        IterationCheckpointPayload,
    )
    from private_gpt.components.engines.chat.models.chat_state import ChatState
    from private_gpt.components.engines.chat.models.execution_hooks import (
        ExecutionHooks,
    )
    from private_gpt.server.chat.chat_service import ChatService


async def execute_chat_start_step(
    *,
    chat_service: ChatService,
    request_data: dict[str, object],
    hooks: ExecutionHooks | None,
    channel: EventChannel,
) -> ChatState:
    request = ResolvedChatRequest.model_validate(request_data)
    engine = chat_service.build_engine()
    return await engine.execute(request, hooks=hooks, channel=channel)


async def execute_chat_resume_step(
    *,
    chat_service: ChatService,
    request_data: dict[str, object],
    pause_type: str,
    tool_results: list[dict[str, object]],
    iteration: int,
    next_block_count: int,
    iteration_state_service: IterationStateService,
    correlation_id: str,
    hooks: ExecutionHooks | None,
    channel: EventChannel,
) -> ChatState:
    request = ResolvedChatRequest.model_validate(request_data)
    engine = chat_service.build_engine()
    saved = await iteration_state_service.load(correlation_id)
    checkpoint_payload: IterationCheckpointPayload | None = None
    if saved is not None:
        checkpoint_payload = saved.checkpoint_payload.model_copy(
            update={
                "tool_responses": [
                    ToolExecutionResponse.model_validate(item) for item in tool_results
                ]
            }
        )
    return await engine.resume(
        pause_type,
        request,
        iteration=iteration,
        next_block_count=next_block_count,
        hooks=hooks,
        checkpoint_payload=checkpoint_payload,
        channel=channel,
    )
