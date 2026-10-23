from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.server.chat_async.chat_async_service import ChatAsyncService


def _service(
    stream_manager: MagicMock | None = None,
    chat_facade: MagicMock | None = None,
) -> ChatAsyncService:
    if stream_manager is None:
        stream_manager = MagicMock()
        stream_manager.stream_exists = AsyncMock(return_value=False)
        stream_manager.cancel_stream = AsyncMock(return_value=False)
    if chat_facade is None:
        chat_facade = MagicMock()

    return ChatAsyncService(stream_manager=stream_manager, chat_facade=chat_facade)


@pytest.mark.anyio
async def test_initiate_chat_stream_starts_engine_event_processing() -> None:
    stream_manager = MagicMock()
    stream_manager.stream_exists = AsyncMock(return_value=False)
    stream_manager.create_and_start_stream = AsyncMock(return_value="msg-1")
    chat_facade = MagicMock()

    async def events():
        if False:
            yield None

    event_generator = events()
    chat_facade.create_chat_event_generator = AsyncMock(return_value=event_generator)

    service = _service(stream_manager=stream_manager, chat_facade=chat_facade)
    request = MagicMock()
    request.messages = []
    request.thinking.enabled = False
    updated_request = MagicMock()
    updated_request.messages = []
    updated_request.thinking.enabled = False
    request.model_copy.return_value = updated_request
    correlation_id = await service.initiate_chat_stream(
        request=request,
        message_id="msg-1",
    )

    assert correlation_id == "msg-1"
    request.context.model_copy.assert_called_once_with(
        update={"correlation_id": "msg-1"}
    )
    chat_facade.create_chat_event_generator.assert_awaited_once_with(
        request=updated_request
    )
    stream_manager.create_and_start_stream.assert_awaited_once_with(
        event_handler=stream_manager.create_and_start_stream.await_args.kwargs[
            "event_handler"
        ],
        stream_type="chat_completion",
        event_generator=event_generator,
        correlation_id="msg-1",
        metadata={"message_count": 0, "thinking_enabled": False},
    )


@pytest.mark.anyio
async def test_initiate_chat_stream_rejects_duplicate_message_id() -> None:
    stream_manager = MagicMock()
    stream_manager.stream_exists = AsyncMock(return_value=True)

    service = _service(stream_manager=stream_manager)

    with pytest.raises(ValueError, match="already exists"):
        await service.initiate_chat_stream(request=MagicMock(), message_id="msg-2")


@pytest.mark.anyio
async def test_cancel_only_delegates_to_stream_manager() -> None:
    stream_manager = MagicMock()
    stream_manager.cancel_stream = AsyncMock(return_value=False)
    chat_facade = MagicMock()
    chat_facade.cancel = AsyncMock(return_value=True)

    service = _service(stream_manager=stream_manager, chat_facade=chat_facade)

    await service.cancel_stream("msg-3")

    stream_manager.cancel_stream.assert_awaited_once_with("msg-3")
    chat_facade.cancel.assert_not_awaited()
