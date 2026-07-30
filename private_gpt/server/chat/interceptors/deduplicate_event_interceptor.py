from collections.abc import Mapping
from typing import Any

from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatResponseLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.events.models import Event, PingEvent


class DeduplicateEventInterceptor(ChatResponseLoopInterceptor):
    """Drop consecutive duplicate streamed events (except pings).

    Some providers re-emit identical ``content_block_delta`` payloads (e.g.
    ``input_json_delta`` with an unchanged ``partial_json_obj``). Those
    duplicates add noise for clients without changing state, so they are
    suppressed. Ping keepalives are always forwarded.

    Equality is checked via pydantic's structural ``__eq__`` (field-by-field
    comparison) rather than ``model_dump_json``, since serializing every
    streamed event just to compare it adds significant latency.
    """

    def __init__(self) -> None:
        self._last_event: Event | None = None

    async def on_iteration_start(self, context: ChatInterceptorContext) -> None:
        self._last_event = None

    async def on_iteration_end(self, context: ChatInterceptorContext) -> None:
        self._last_event = None

    async def intercept_event(
        self,
        event: Event,
        context: ChatInterceptorContext,
    ) -> Event | None:
        if isinstance(event, PingEvent):
            return event

        if event == self._last_event:
            return None

        self._last_event = event
        return event

    def model_copy(
        self, *, update: Mapping[str, Any] | None | None = None, deep: bool = False
    ) -> "DeduplicateEventInterceptor":
        return DeduplicateEventInterceptor()
