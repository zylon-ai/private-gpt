from collections.abc import Mapping
from typing import Any

from private_gpt.components.engines.chat_loop.interceptors.chat_loop_interceptor import (
    ChatResponseLoopInterceptor,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_interceptor_context import (
    ChatLoopInterceptorContext,
)
from private_gpt.events.models import (
    Event,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
)


class EnsureIndexIsRefreshedInterceptor(ChatResponseLoopInterceptor):

    _current_index: int = 0
    _block_id_map: dict[str, int] | None = None

    async def on_iteration_start(self, context: ChatLoopInterceptorContext) -> None:
        self._block_id_map = {}

    async def intercept_event(
        self, event: Event, context: ChatLoopInterceptorContext
    ) -> Event | None:
        if self._block_id_map is None:
            raise ValueError("Block ID map is not initialized. This should not happen.")
        match event:
            case RawContentBlockStartEvent(block_id=block_id):
                if block_id in self._block_id_map:
                    raise ValueError(
                        f"Received duplicate ContentBlockStart for blockId {block_id}"
                    )
                self._block_id_map[block_id] = self._current_index
                self._current_index += 1
                return event.model_copy(update={"index": self._block_id_map[block_id]})

            case RawContentBlockDeltaEvent(
                block_id=block_id
            ) | RawContentBlockStopEvent(block_id=block_id):
                if block_id not in self._block_id_map:
                    raise ValueError(
                        f"Received {type(event).__name__} for unknown blockId {block_id}"
                    )
                return event.model_copy(update={"index": self._block_id_map[block_id]})

            case _:
                return event

    def model_copy(
        self, *, update: Mapping[str, Any] | None | None = None, deep: bool = False
    ) -> "EnsureIndexIsRefreshedInterceptor":
        # Return a new instance with the same logic but reset state
        return EnsureIndexIsRefreshedInterceptor()
