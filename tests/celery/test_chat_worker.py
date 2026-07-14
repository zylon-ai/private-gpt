import asyncio
from typing import Any

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
