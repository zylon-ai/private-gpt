from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from private_gpt.arq.iteration_state import IterationStateService
    from private_gpt.components.engines.chat.models.chat_state import (
        ChatState,
    )


class IterationScheduler:
    """Transport-agnostic: persists iteration state when engine returns WAITING."""

    def __init__(
        self,
        *,
        iteration_state_service: IterationStateService,
        correlation_id: str,
        request_data: dict[str, Any],
        stream_type: str,
        metadata: dict[str, Any],
    ) -> None:
        self._iteration_state_service = iteration_state_service
        self._correlation_id = correlation_id
        self._request_data = request_data
        self._stream_type = stream_type
        self._metadata = metadata
        self._was_waiting = False

    async def on_waiting(self, state: ChatState) -> None:
        from private_gpt.arq.iteration_state import IterationContext
        from private_gpt.components.engines.chat.async_chat_engine import (
            IterationCheckpointPayload,
        )

        self._was_waiting = True
        await self._iteration_state_service.save(
            IterationContext(
                correlation_id=self._correlation_id,
                request_data=state.input.request.model_dump(mode="json"),
                stream_type=self._stream_type,
                metadata=self._metadata,
                iteration=state.runtime.iteration,
                checkpoint=state.output.pause_type,
                checkpoint_payload=IterationCheckpointPayload(
                    pending_async_tools=state.output.pending_async_tools,
                    pending_external_tool_calls=list(
                        state.output.pending_external_tool_calls
                    ),
                ),
                next_block_count=state.runtime.next_block_count,
            )
        )

    @property
    def was_waiting(self) -> bool:
        return self._was_waiting
