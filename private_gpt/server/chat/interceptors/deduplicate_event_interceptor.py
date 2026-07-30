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
    """

    def __init__(self) -> None:
        self._last_fingerprint: str | None = None

    async def on_iteration_start(self, context: ChatInterceptorContext) -> None:
        self._last_fingerprint = None

    async def on_iteration_end(self, context: ChatInterceptorContext) -> None:
        self._last_fingerprint = None

    async def intercept_event(
        self,
        event: Event,
        context: ChatInterceptorContext,
    ) -> Event | None:
        if isinstance(event, PingEvent):
            return event

        fingerprint = event.model_dump_json()
        if fingerprint == self._last_fingerprint:
            return None

        self._last_fingerprint = fingerprint
        return event

    def model_copy(
        self, *, update: Mapping[str, Any] | None | None = None, deep: bool = False
    ) -> "DeduplicateEventInterceptor":
        return DeduplicateEventInterceptor()
