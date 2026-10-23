import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.events.models import PingEvent
from private_gpt.server.chat.chat_facade import ChatFacadeService


@pytest.mark.asyncio
async def test_event_generator_close_propagates_to_engine_generator() -> None:
    closed = asyncio.Event()

    async def engine_events():
        try:
            yield PingEvent()
            await asyncio.Event().wait()
        finally:
            closed.set()

    chat_service = MagicMock()
    chat_service.stream_chat = AsyncMock(
        return_value=SimpleNamespace(events=engine_events())
    )
    facade = ChatFacadeService(
        chat_service=chat_service,
        stream_manager=MagicMock(),
    )
    event_generator = await facade.create_chat_event_generator(request=MagicMock())

    await anext(event_generator)
    await event_generator.aclose()

    assert closed.is_set()
