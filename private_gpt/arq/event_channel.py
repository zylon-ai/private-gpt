from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from private_gpt.components.engines.chat.async_chat_engine import EventChannel

if TYPE_CHECKING:
    from private_gpt.arq.iteration_state import IterationStateService
    from private_gpt.events.event_serializer import StreamingEventHandler
    from private_gpt.events.models import Event


class RedisEventChannel(EventChannel):
    """Writes events to Redis via IterationStateService."""

    def __init__(
        self,
        service: IterationStateService,
        correlation_id: str,
        event_handler: StreamingEventHandler,
    ) -> None:
        self._service = service
        self._cid = correlation_id
        self._handler = event_handler
        self._pending: list[asyncio.Task[None]] = []

    def emit(self, event: Event) -> None:
        task = asyncio.create_task(
            self._service.append_event(self._cid, self._handler.serialize(event))
        )
        self._pending.append(task)

    async def close(self) -> None:
        if self._pending:
            await asyncio.gather(*self._pending)
            self._pending.clear()
        await self._service.append_done(self._cid)
