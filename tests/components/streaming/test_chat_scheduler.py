from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.chat.input_models import MessageInput
from private_gpt.components.streaming.providers.models import StreamStatus
from private_gpt.components.streaming.tasks.chat_scheduler import (
    ArqChatScheduler,
    ChatSchedulerFactory,
    LocalChatScheduler,
)
from private_gpt.server.chat.chat_models import ChatBody


def _chat_body() -> ChatBody:
    return ChatBody(messages=[MessageInput(role="user", content="hello")])


def _make_stream_component(correlation_id: str = "msg-1") -> MagicMock:
    stream_service = AsyncMock()
    stream_service.create_stream = AsyncMock(return_value=correlation_id)
    stream_service.update_stream_status = AsyncMock()
    stream_component = MagicMock()
    stream_component.stream = stream_service
    return stream_component


@pytest.mark.anyio
async def test_arq_chat_scheduler_create_enqueues_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue_chat_job = AsyncMock()
    monkeypatch.setattr(
        "private_gpt.components.streaming.tasks.chat_scheduler.enqueue_chat_job",
        enqueue_chat_job,
    )

    stream_component = _make_stream_component("msg-arq-1")
    scheduler = ArqChatScheduler(stream_component=stream_component)

    correlation_id = await scheduler.create(_chat_body(), message_id="msg-arq-1")

    assert correlation_id == "msg-arq-1"
    enqueue_chat_job.assert_awaited_once()


@pytest.mark.anyio
async def test_arq_chat_scheduler_create_marks_error_on_enqueue_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "private_gpt.components.streaming.tasks.chat_scheduler.enqueue_chat_job",
        AsyncMock(side_effect=RuntimeError("redis unavailable")),
    )

    stream_component = _make_stream_component("msg-arq-2")
    scheduler = ArqChatScheduler(stream_component=stream_component)

    with pytest.raises(RuntimeError, match="redis unavailable"):
        await scheduler.create(_chat_body(), message_id="msg-arq-2")

    stream_component.stream.update_stream_status.assert_awaited_once_with(
        "msg-arq-2",
        StreamStatus.ERROR,
        error_message="redis unavailable",
        metadata={"message_count": 1},
    )


@pytest.mark.anyio
async def test_arq_chat_scheduler_cancel_aborts_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    abort_chat_job = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "private_gpt.components.streaming.tasks.chat_scheduler.abort_chat_job",
        abort_chat_job,
    )

    scheduler = ArqChatScheduler(stream_component=_make_stream_component())
    cancelled = await scheduler.cancel("msg-arq-3")

    assert cancelled is True
    abort_chat_job.assert_awaited_once_with(correlation_id="msg-arq-3")


@pytest.mark.anyio
async def test_local_chat_scheduler_create_starts_task() -> None:
    chat_service = AsyncMock()
    chat_request_mapper = AsyncMock()
    chat_request_mapper.create_request_from_body.return_value = MagicMock()

    start_gen = AsyncMock()
    start_state = MagicMock()
    start_state.runtime.iteration = 0
    start_state.runtime.next_block_count = 0
    start_state.input.request.model_dump.return_value = {}
    start_gen.final_state_task = AsyncMock(return_value=start_state)()
    chat_service.start_chat.return_value = start_gen

    iter_gen = AsyncMock()
    iter_state = MagicMock()
    from private_gpt.components.engines.chat.models.chat_state import (
        ChatStatus,
    )

    iter_state.output.status = ChatStatus.COMPLETED
    iter_state.output.stop_reason = "end_turn"
    iter_state.runtime.iteration = 1
    iter_state.runtime.next_block_count = 0
    iter_state.input.request.model_dump.return_value = {}
    iter_gen.final_state_task = AsyncMock(return_value=iter_state)()
    chat_service.run_iteration.return_value = iter_gen

    close_gen = AsyncMock()
    close_state = MagicMock()
    close_gen.final_state_task = AsyncMock(return_value=close_state)()
    chat_service.close_chat.return_value = close_gen

    stream_component = _make_stream_component("msg-local-1")
    stream_processor = MagicMock()
    stream_processor.process_stream = AsyncMock()
    stream_processor.stream_service = stream_component.stream
    task_manager = AsyncMock()
    task_manager.create_task = AsyncMock()
    iteration_state_service = AsyncMock()

    scheduler = LocalChatScheduler(
        chat_service=chat_service,
        chat_request_mapper=chat_request_mapper,
        stream_component=stream_component,
        stream_processor=stream_processor,
        task_manager=task_manager,
        iteration_state_service=iteration_state_service,
    )

    correlation_id = await scheduler.create(_chat_body(), message_id="msg-local-1")

    assert correlation_id == "msg-local-1"
    task_manager.create_task.assert_awaited_once()


@pytest.mark.anyio
async def test_local_chat_scheduler_cancel_cancels_task() -> None:
    task_manager = AsyncMock()
    task_manager.cancel_task = AsyncMock(return_value=True)

    scheduler = LocalChatScheduler(
        chat_service=AsyncMock(),
        chat_request_mapper=AsyncMock(),
        stream_component=_make_stream_component(),
        stream_processor=MagicMock(),
        task_manager=task_manager,
        iteration_state_service=AsyncMock(),
    )

    cancelled = await scheduler.cancel("msg-local-2")

    assert cancelled is True
    task_manager.cancel_task.assert_awaited_once_with("msg-local-2")


def test_chat_scheduler_factory_selects_mode() -> None:
    factory = ChatSchedulerFactory(
        settings=SimpleNamespace(
            scheduler=SimpleNamespace(chat=SimpleNamespace(mode="arq"))
        ),
        local=MagicMock(),
        arq=ArqChatScheduler(stream_component=_make_stream_component()),
    )

    assert isinstance(factory.get(), ArqChatScheduler)
