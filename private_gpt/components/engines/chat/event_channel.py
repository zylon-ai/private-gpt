from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from private_gpt.components.engines.chat.async_chat_engine import EventChannel

if TYPE_CHECKING:
    from private_gpt.components.engines.chat.event_broker import EngineEventBroker
    from private_gpt.events.models import Event


class BrokerEventChannel(EventChannel):
    """Ordered EventChannel adapter backed by an EngineEventBroker."""

    def __init__(self, broker: EngineEventBroker, execution_id: str) -> None:
        self._broker = broker
        self._execution_id = execution_id
        self._tail: asyncio.Future[None] | None = None

    def emit(self, event: Event) -> None:
        previous = self._tail

        async def _publish() -> None:
            if previous is not None:
                await previous
            await self._broker.publish(self._execution_id, event)

        self._tail = asyncio.create_task(_publish())

    async def flush(self) -> None:
        if self._tail is not None:
            await self._tail

    async def close(self) -> None:
        await self.flush()
