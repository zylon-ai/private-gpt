from unittest.mock import AsyncMock

import pytest

from private_gpt.chat.input_models import MessageInput
from private_gpt.components.streaming.providers.in_memory_stream_service import (
    InMemoryStreamService,
)
from private_gpt.server.chat.chat_models import ChatBody
from private_gpt.server.chat_async.chat_async_service import ChatAsyncService


class _StreamManagerStub:
    def __init__(self, stream_service: InMemoryStreamService) -> None:
        self.stream_service = stream_service
        self.cancel_stream = AsyncMock(return_value=False)
        self.stream_exists = AsyncMock(return_value=False)


class _ChatSchedulerFactoryStub:
    def __init__(self, scheduler: object) -> None:
        self._scheduler = scheduler

    def get(self) -> object:
        return self._scheduler


class _ChatSchedulerStub:
    def __init__(self) -> None:
        self.create = AsyncMock(return_value="msg-1")


def _service(
    stream_service: InMemoryStreamService,
    scheduler: _ChatSchedulerStub | None = None,
) -> ChatAsyncService:
    scheduler = scheduler or _ChatSchedulerStub()
    return ChatAsyncService(
        stream_manager=_StreamManagerStub(stream_service),  # type: ignore[arg-type]
        chat_scheduler_factory=_ChatSchedulerFactoryStub(scheduler),  # type: ignore[arg-type]
    )


def _chat_body() -> ChatBody:
    return ChatBody(messages=[MessageInput(role="user", content="hello")])


@pytest.mark.anyio
async def test_initiate_chat_stream_delegates_to_scheduler() -> None:
    stream_service = InMemoryStreamService()
    scheduler = _ChatSchedulerStub()
    service = _service(stream_service, scheduler)

    correlation_id = await service.initiate_chat_stream(
        body=_chat_body(),
        message_id="msg-1",
    )

    assert correlation_id == "msg-1"
    scheduler.create.assert_awaited_once_with(body=_chat_body(), message_id="msg-1")


@pytest.mark.anyio
async def test_initiate_chat_stream_rejects_duplicate_message_id() -> None:
    stream_service = InMemoryStreamService()
    scheduler = _ChatSchedulerStub()
    service = _service(stream_service, scheduler)
    service.stream_manager.stream_exists.return_value = True

    with pytest.raises(ValueError, match="already exists"):
        await service.initiate_chat_stream(body=_chat_body(), message_id="msg-2")

    scheduler.create.assert_not_awaited()


@pytest.mark.anyio
async def test_worker_cancel_marks_stream_cancelled() -> None:
    stream_service = InMemoryStreamService()
    service = _service(stream_service)

    await service.cancel_stream("msg-3")

    service.stream_manager.cancel_stream.assert_awaited_once_with("msg-3")


@pytest.mark.anyio
async def test_arq_cancel_delegates_to_stream_manager() -> None:
    stream_service = InMemoryStreamService()
    service = _service(stream_service)

    await service.cancel_stream("msg-4")

    service.stream_manager.cancel_stream.assert_awaited_once_with("msg-4")
