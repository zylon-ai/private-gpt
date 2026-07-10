from collections.abc import Mapping
from typing import Any, Literal

from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatResponseLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.events.models import (
    Event,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
)


class FilterZylonInterceptor(ChatResponseLoopInterceptor):
    def __init__(self) -> None:
        self._active_blocks: set[str] = set()

    async def on_iteration_start(self, context: ChatInterceptorContext) -> None:
        self._active_blocks.clear()

    async def on_iteration_end(self, context: ChatInterceptorContext) -> None:
        self._active_blocks.clear()

    async def intercept_event(
        self,
        event: Event,
        context: ChatInterceptorContext,
    ) -> Event | None:
        response_format: Literal["zylon", "anthropic"] = (
            "zylon"
            if context.state.input.request.system.extensions.zylon_enabled
            else "anthropic"
        )
        if not event.prune_content_block_by_response_mode(response_format):
            return None

        if isinstance(event, RawContentBlockStartEvent):
            self._active_blocks.add(event.block_id)
            return event

        if isinstance(event, RawContentBlockStopEvent):
            if event.block_id not in self._active_blocks:
                return None
            self._active_blocks.discard(event.block_id)

        return event

    def model_copy(
        self, *, update: Mapping[str, Any] | None | None = None, deep: bool = False
    ) -> "FilterZylonInterceptor":
        # Return a new instance with the same logic but reset state
        return FilterZylonInterceptor()
