from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.chat.input_models import MessageInput
from private_gpt.components.streaming.providers.models import StreamStatus
from private_gpt.components.streaming.tasks.chat_scheduler import (
    ArqChatScheduler,
    CeleryChatScheduler,
    ChatSchedulerFactory,
    LocalChatScheduler,
)
from private_gpt.server.chat.chat_models import ChatBody


class _StreamManagerStub:
    def __init__(self) -> None:
        self.stream_service = AsyncMock()
        self.create_and_start_stream = AsyncMock(return_value="msg-local")


def _chat_body() -> ChatBody:
    return ChatBody(messages=[MessageInput(role="user", content="hello")])


@pytest.mark.anyio
async def test_celery_chat_scheduler_create_dispatches_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch_task = MagicMock()
    monkeypatch.setattr(
        "private_gpt.components.streaming.tasks.chat_scheduler.dispatch_task",
        dispatch_task,
    )

    stream_manager = _StreamManagerStub()
    stream_manager.stream_service.create_stream.return_value = "msg-1"
    scheduler = CeleryChatScheduler(
        settings=SimpleNamespace(
            scheduler=SimpleNamespace(chat=SimpleNamespace(celery_queue="chat"))
        ),
        stream_manager=stream_manager,
    )

    correlation_id = await scheduler.create(_chat_body(), message_id="msg-1")

    assert correlation_id == "msg-1"
    dispatch_task.assert_called_once()


@pytest.mark.anyio
async def test_celery_chat_scheduler_create_marks_error_on_dispatch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "private_gpt.components.streaming.tasks.chat_scheduler.dispatch_task",
        MagicMock(side_effect=RuntimeError("broker unavailable")),
    )

    stream_manager = _StreamManagerStub()
    stream_manager.stream_service.create_stream.return_value = "msg-2"
    scheduler = CeleryChatScheduler(
        settings=SimpleNamespace(
            scheduler=SimpleNamespace(chat=SimpleNamespace(celery_queue="chat"))
        ),
        stream_manager=stream_manager,
    )

    with pytest.raises(RuntimeError, match="broker unavailable"):
        await scheduler.create(_chat_body(), message_id="msg-2")

    stream_manager.stream_service.update_stream_status.assert_awaited_once_with(
        "msg-2",
        StreamStatus.ERROR,
        error_message="broker unavailable",
        metadata={"message_count": 1},
    )


@pytest.mark.anyio
async def test_celery_chat_scheduler_cancel_revokes_without_terminate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    celery_app = MagicMock()
    monkeypatch.setattr("private_gpt.celery.celery.celery_app", celery_app)

    scheduler = CeleryChatScheduler(
        settings=SimpleNamespace(),
        stream_manager=_StreamManagerStub(),
    )

    cancelled = await scheduler.cancel("msg-1")

    assert cancelled is True
    celery_app.control.revoke.assert_called_once_with("msg-1", terminate=False)


@pytest.mark.anyio
async def test_arq_chat_scheduler_create_enqueues_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue_chat_job = AsyncMock()
    monkeypatch.setattr(
        "private_gpt.components.streaming.tasks.chat_scheduler.enqueue_chat_job",
        enqueue_chat_job,
    )

    stream_manager = _StreamManagerStub()
    stream_manager.stream_service.create_stream.return_value = "msg-arq-1"
    scheduler = ArqChatScheduler(stream_manager=stream_manager)

    correlation_id = await scheduler.create(_chat_body(), message_id="msg-arq-1")

    assert correlation_id == "msg-arq-1"
    enqueue_chat_job.assert_awaited_once()


@pytest.mark.anyio
async def test_arq_chat_scheduler_cancel_aborts_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    abort_chat_job = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "private_gpt.components.streaming.tasks.chat_scheduler.abort_chat_job",
        abort_chat_job,
    )

    scheduler = ArqChatScheduler(stream_manager=_StreamManagerStub())

    cancelled = await scheduler.cancel("msg-2")

    assert cancelled is True
    abort_chat_job.assert_awaited_once_with(correlation_id="msg-2")


@pytest.mark.anyio
async def test_local_chat_scheduler_create_starts_stream() -> None:
    chat_facade = AsyncMock()
    chat_facade.create_chat_event_generator.return_value = SimpleNamespace(events="gen")
    chat_request_mapper = AsyncMock()
    chat_request_mapper.create_request_from_body.return_value = "request"
    stream_manager = _StreamManagerStub()
    scheduler = LocalChatScheduler(
        chat_facade=chat_facade,
        chat_request_mapper=chat_request_mapper,
        stream_manager=stream_manager,
    )

    correlation_id = await scheduler.create(_chat_body(), message_id="msg-local")

    assert correlation_id == "msg-local"
    stream_manager.create_and_start_stream.assert_awaited_once()


@pytest.mark.anyio
async def test_local_chat_scheduler_cancel_is_noop() -> None:
    scheduler = LocalChatScheduler(
        chat_facade=AsyncMock(),
        chat_request_mapper=AsyncMock(),
        stream_manager=_StreamManagerStub(),
    )

    cancelled = await scheduler.cancel("msg-3")

    assert cancelled is False


def test_chat_scheduler_factory_selects_mode() -> None:
    factory = ChatSchedulerFactory(
        settings=SimpleNamespace(
            scheduler=SimpleNamespace(chat=SimpleNamespace(mode="arq"))
        ),
        local=MagicMock(),
        celery=MagicMock(),
        arq=ArqChatScheduler(stream_manager=_StreamManagerStub()),
    )

    assert isinstance(factory.get(), ArqChatScheduler)
