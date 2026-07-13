import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from injector import Injector

from private_gpt.components.streaming.tasks.chat_scheduler import (
    ArqChatScheduler,
    ChatSchedulerFactory,
    LocalChatScheduler,
)


@pytest.fixture
def injector() -> Injector:
    return Injector()


@pytest.mark.anyio
async def test_arq_chat_scheduler_cancel_aborts_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    abort_chat_job = AsyncMock(return_value=True)
    monkeypatch.setattr(
        "private_gpt.components.streaming.tasks.chat_scheduler.abort_chat_job",
        abort_chat_job,
    )

    scheduler = ArqChatScheduler()
    cancelled = await scheduler.cancel("msg-arq-3")

    assert cancelled is True
    abort_chat_job.assert_awaited_once_with(correlation_id="msg-arq-3")


@pytest.mark.anyio
async def test_local_chat_scheduler_cancel_cancels_task() -> None:
    async def _work() -> None:
        await asyncio.sleep(100)

    task = asyncio.create_task(_work(), name="chat_msg-local-2")

    scheduler = LocalChatScheduler()
    cancelled = await scheduler.cancel("msg-local-2")

    assert cancelled is True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.anyio
async def test_local_chat_scheduler_cancel_returns_false_when_no_task() -> None:
    scheduler = LocalChatScheduler()
    cancelled = await scheduler.cancel("nonexistent-msg")
    assert cancelled is False


def test_chat_scheduler_factory_selects_local_mode(injector: Injector) -> None:
    factory = ChatSchedulerFactory(
        settings=SimpleNamespace(
            scheduler=SimpleNamespace(chat=SimpleNamespace(mode="local"))
        ),
        injector=injector,
    )
    assert isinstance(factory.get(), LocalChatScheduler)


def test_chat_scheduler_factory_selects_arq_mode(injector: Injector) -> None:
    factory = ChatSchedulerFactory(
        settings=SimpleNamespace(
            scheduler=SimpleNamespace(chat=SimpleNamespace(mode="arq"))
        ),
        injector=injector,
    )
    assert isinstance(factory.get(), ArqChatScheduler)


def test_chat_scheduler_factory_raises_on_unknown_mode(injector: Injector) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        ChatSchedulerFactory(
            settings=SimpleNamespace(
                scheduler=SimpleNamespace(chat=SimpleNamespace(mode="unknown"))
            ),
            injector=injector,
        )


@pytest.mark.anyio
async def test_arq_chat_scheduler_cancel_returns_false_when_abort_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    abort_chat_job = AsyncMock(return_value=False)
    monkeypatch.setattr(
        "private_gpt.components.streaming.tasks.chat_scheduler.abort_chat_job",
        abort_chat_job,
    )

    scheduler = ArqChatScheduler()
    cancelled = await scheduler.cancel("msg-arq-not-found")

    assert cancelled is False
    abort_chat_job.assert_awaited_once_with(correlation_id="msg-arq-not-found")


@pytest.mark.anyio
async def test_local_chat_scheduler_cancel_marks_task_as_cancelled() -> None:
    async def _work() -> None:
        await asyncio.sleep(100)

    task = asyncio.create_task(_work(), name="chat_msg-local-mark")

    scheduler = LocalChatScheduler()
    result = await scheduler.cancel("msg-local-mark")
    assert result is True

    with pytest.raises(asyncio.CancelledError):
        await task

    assert task.cancelled()


@pytest.mark.anyio
async def test_local_chat_scheduler_cancel_only_targets_the_matching_task() -> None:
    async def _work() -> None:
        await asyncio.sleep(100)

    task_target = asyncio.create_task(_work(), name="chat_target-cid")
    task_other = asyncio.create_task(_work(), name="chat_other-cid")

    scheduler = LocalChatScheduler()
    result = await scheduler.cancel("target-cid")
    assert result is True

    with pytest.raises(asyncio.CancelledError):
        await task_target
    assert task_target.cancelled()

    assert not task_other.done()
    task_other.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task_other
