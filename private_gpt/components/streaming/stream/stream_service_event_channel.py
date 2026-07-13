from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from private_gpt.components.engines.chat.async_chat_engine import EventChannel

if TYPE_CHECKING:
    from private_gpt.components.streaming.providers.stream_service import StreamService
    from private_gpt.events.event_serializer import StreamingEventHandler
    from private_gpt.events.models import Event


class StreamServiceEventChannel(EventChannel):
    """Writes engine events directly to the configured stream service."""

    def __init__(
        self,
        stream_service: StreamService,
        correlation_id: str,
        event_handler: StreamingEventHandler,
    ) -> None:
        self._stream_service = stream_service
        self._cid = correlation_id
        self._handler = event_handler
        self._pending: list[asyncio.Task[None]] = []

    def emit(self, event: Event) -> None:
        async def _write() -> None:
            event_data = self._handler.serialize(event)
            await self._stream_service.push_event(self._cid, event_data)
            status = await self._handler.get_current_status(event)
            if status is not None:
                await self._stream_service.update_stream_status(self._cid, status)

        task = asyncio.create_task(_write())
        self._pending.append(task)

    async def close(self) -> None:
        if self._pending:
            await asyncio.gather(*self._pending)
            self._pending.clear()
