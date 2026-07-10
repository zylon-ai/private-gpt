import datetime
from collections.abc import Mapping
from typing import Any

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


class EnsureTimestampInContentBlocksInterceptor(ChatResponseLoopInterceptor):
    async def intercept_event(
        self, event: Event, context: ChatInterceptorContext
    ) -> Event | None:
        match event:
            case RawContentBlockStartEvent():
                event.content_block.start_timestamp = datetime.datetime.now(
                    datetime.UTC
                )
                return event

            case RawContentBlockStopEvent():
                event.stop_timestamp = datetime.datetime.now(datetime.UTC)
                return event
            case _:
                return event

    def model_copy(
        self, *, update: Mapping[str, Any] | None | None = None, deep: bool = False
    ) -> "EnsureTimestampInContentBlocksInterceptor":
        # Return a new instance with the same logic but reset state
        return EnsureTimestampInContentBlocksInterceptor()
