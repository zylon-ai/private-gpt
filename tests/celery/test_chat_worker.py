import asyncio
from typing import Any
from unittest.mock import Mock

import pytest

from private_gpt.celery.base import StatefulBackgroundTask


def test_stateful_task_reuses_one_event_loop() -> None:
    class LoopIdTask(StatefulBackgroundTask):
        name = "test_chat_loop_id_task"

        @classmethod
        def warm_up(cls) -> None:
            cls._ensure_runtime()
            cls.run_coroutine(cls._create_test_injector())
            cls._warmed = True

        @classmethod
        async def _create_test_injector(cls) -> None:
            from private_gpt.di import create_loop_injector

            create_loop_injector()

        async def run(self, *args: Any, **kwargs: Any) -> int:
            return id(asyncio.get_running_loop())

    try:
        task = LoopIdTask()

        first_loop_id = task()
        second_loop_id = task()

        assert first_loop_id == second_loop_id
    finally:
        LoopIdTask.shutdown_runtime()


async def test_stateful_worker_uses_explicit_warm_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injector = Mock()
    warm = Mock()
    monkeypatch.setenv("PGPT_STATEFUL_WORKER_TYPE", "stateful-type")
    monkeypatch.setenv("PGPT_WORKER_WARM_PROFILE", "tools")
    monkeypatch.setattr(
        "private_gpt.celery.base.create_loop_injector",
        Mock(return_value=injector),
    )
    monkeypatch.setattr("private_gpt.eager_loading.warm", warm)

    await StatefulBackgroundTask._warm_async()

    warm.assert_called_once_with(injector, profile="tools")


def test_stateful_worker_discards_inherited_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inherited_loop = Mock()
    inherited_thread = Mock()
    discard_inherited_injectors = Mock()
    monkeypatch.setattr(
        "private_gpt.celery.base.discard_inherited_injectors",
        discard_inherited_injectors,
    )
    StatefulBackgroundTask._loop = inherited_loop
    StatefulBackgroundTask._thread = inherited_thread
    StatefulBackgroundTask._warmed = True

    StatefulBackgroundTask.reset_after_fork()

    assert StatefulBackgroundTask._loop is None
    assert StatefulBackgroundTask._thread is None
    assert StatefulBackgroundTask._warmed is False
    discard_inherited_injectors.assert_called_once_with(inherited_loop)


async def test_stateful_worker_requires_explicit_warm_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PGPT_WORKER_WARM_PROFILE", raising=False)

    with pytest.raises(ValueError, match="PGPT_WORKER_WARM_PROFILE"):
        await StatefulBackgroundTask._warm_async()
