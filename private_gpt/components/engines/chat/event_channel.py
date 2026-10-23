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
        self._pending: list[Event] = []
        self._drain: asyncio.Task[None] | None = None

    def emit(self, event: Event) -> None:
        # One drain task publishes everything emitted meanwhile as a single batch.
        self._pending.append(event)
        if self._drain is None or (
            self._drain.done() and self._drain.exception() is None
        ):
            self._drain = asyncio.create_task(self._publish_pending())

    async def _publish_pending(self) -> None:
        while self._pending:
            batch, self._pending = self._pending, []
            await self._broker.publish_many(self._execution_id, batch)

    async def flush(self) -> None:
        while self._drain is not None:
            await self._drain
            if not self._pending:
                return
            self._drain = asyncio.create_task(self._publish_pending())

    async def close(self) -> None:
        await self.flush()
