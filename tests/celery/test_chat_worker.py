import asyncio
from typing import Any

from private_gpt.celery.base import ChatBackgroundTask
from private_gpt.celery.celery import celery_app


def test_chat_task_is_registered_by_default_imports() -> None:
    celery_app.loader.import_default_modules()

    assert "private_gpt.chat.run" in celery_app.tasks


def test_chat_background_task_reuses_one_event_loop() -> None:
    class LoopIdTask(ChatBackgroundTask):
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
