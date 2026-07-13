from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.events.models import Event
from private_gpt.server.chat_async.chat_async_service import ChatAsyncService


async def _empty_gen() -> AsyncGenerator[Event, None]:
    return
    yield  # make it an async generator


def _service(
    stream_manager: MagicMock | None = None,
    chat_facade: MagicMock | None = None,
) -> ChatAsyncService:
    if stream_manager is None:
        stream_manager = MagicMock()
        stream_manager.stream_exists = AsyncMock(return_value=False)
        stream_manager.create_and_start_stream = AsyncMock(return_value="msg-1")
        stream_manager.cancel_stream = AsyncMock(return_value=False)

    if chat_facade is None:
        chat_facade = MagicMock()
        chat_facade.create_chat_event_generator = AsyncMock(return_value=_empty_gen())

    return ChatAsyncService(
        chat_facade=chat_facade,
        stream_manager=stream_manager,
    )


@pytest.mark.anyio
async def test_initiate_chat_stream_delegates_to_scheduler() -> None:
    stream_manager = MagicMock()
    stream_manager.stream_exists = AsyncMock(return_value=False)
    stream_manager.create_and_start_stream = AsyncMock(return_value="msg-1")

    chat_facade = MagicMock()
    chat_facade.create_chat_event_generator = AsyncMock(return_value=_empty_gen())

    service = _service(stream_manager=stream_manager, chat_facade=chat_facade)

    correlation_id = await service.initiate_chat_stream(
        request=MagicMock(),
        message_id="msg-1",
    )

    assert correlation_id == "msg-1"
    stream_manager.create_and_start_stream.assert_awaited_once()


@pytest.mark.anyio
async def test_initiate_chat_stream_rejects_duplicate_message_id() -> None:
    stream_manager = MagicMock()
    stream_manager.stream_exists = AsyncMock(return_value=True)

    service = _service(stream_manager=stream_manager)

    with pytest.raises(ValueError, match="already exists"):
        await service.initiate_chat_stream(request=MagicMock(), message_id="msg-2")


@pytest.mark.anyio
async def test_worker_cancel_marks_stream_cancelled() -> None:
    stream_manager = MagicMock()
    stream_manager.cancel_stream = AsyncMock(return_value=False)

    service = _service(stream_manager=stream_manager)

    await service.cancel_stream("msg-3")

    stream_manager.cancel_stream.assert_awaited_once_with("msg-3")


@pytest.mark.anyio
async def test_arq_cancel_delegates_to_stream_manager() -> None:
    stream_manager = MagicMock()
    stream_manager.cancel_stream = AsyncMock(return_value=False)

    service = _service(stream_manager=stream_manager)

    await service.cancel_stream("msg-4")

    stream_manager.cancel_stream.assert_awaited_once_with("msg-4")
