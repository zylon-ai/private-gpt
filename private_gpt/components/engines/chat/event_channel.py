from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

from private_gpt.components.engines.chat.async_chat_engine import EventChannel

if TYPE_CHECKING:
    from private_gpt.components.engines.chat.event_broker import EngineEventBroker
    from private_gpt.events.models import Event
    from private_gpt.observability.message import MessageFolder


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


class RecordingEventChannel(EventChannel):
    """Forward events unchanged while folding them into a message for tracing."""

    def __init__(self, inner: EventChannel, folder: MessageFolder) -> None:
        self._inner = inner
        self.folder = folder

    def emit(self, event: Event) -> None:
        with contextlib.suppress(Exception):
            self.folder.add(event)
        self._inner.emit(event)

    async def flush(self) -> None:
        await self._inner.flush()

    async def close(self) -> None:
        await self._inner.close()
