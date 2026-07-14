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
            cls._warmed = True

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
        "private_gpt.celery.base.get_global_injector",
        Mock(return_value=injector),
    )
    monkeypatch.setattr("private_gpt.eager_loading.warm", warm)

    await StatefulBackgroundTask._warm_async()

    warm.assert_called_once_with(injector, profile="tools")


async def test_stateful_worker_requires_explicit_warm_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PGPT_WORKER_WARM_PROFILE", raising=False)

    with pytest.raises(ValueError, match="PGPT_WORKER_WARM_PROFILE"):
        await StatefulBackgroundTask._warm_async()
