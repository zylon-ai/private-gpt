from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from injector import inject, singleton

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


class BaseToolScheduler(ABC):
    @abstractmethod
    async def execute(
        self,
        tool_name: str,
        chat_priority: int | None,
        func: Callable[[], Awaitable[Any]],
    ) -> Any:
        ...


@singleton
class LocalToolScheduler(BaseToolScheduler):
    """Execute tools in-process (no worker dispatch)."""

    async def execute(
        self,
        tool_name: str,
        chat_priority: int | None,
        func: Callable[[], Awaitable[Any]],
    ) -> Any:
        return await func()


@singleton
class CeleryToolScheduler(BaseToolScheduler):
    """Dispatch tool calls to a dedicated Celery tools worker.

    When ``scheduler.tools.mode`` is ``"celery"``, the chat worker sends tool
    calls to the ``tools`` queue instead of executing them in-process.
    """

    @inject
    def __init__(self, settings: Settings) -> None:
        from private_gpt.celery.celery import celery_app

        self._celery_app = celery_app
        self._tools_queue = settings.scheduler.tools.celery_queue

    async def execute(
        self,
        tool_name: str,
        chat_priority: int | None,
        func: Callable[[], Awaitable[Any]],
    ) -> Any:
        keywords = getattr(func, "keywords", {})
        tool_id = keywords.get("tool_id", "")
        tool_kwargs = keywords.get("tool_kwargs", {})

        result = self._celery_app.send_task(
            "private_gpt.tools.run",
            args=[tool_name, chat_priority, tool_id, tool_kwargs],
            queue=self._tools_queue,
        )

        while not result.ready():
            await asyncio.sleep(0.1)

        if result.failed():
            raise result.result

        return result.result


@singleton
class ToolSchedulerFactory:
    @inject
    def __init__(
        self,
        settings: Settings,
        local: LocalToolScheduler,
        celery: CeleryToolScheduler,
    ) -> None:
        self._scheduler: BaseToolScheduler = {
            "local": local,
            "celery": celery,
        }[settings.scheduler.tools.mode]

    def get(self) -> BaseToolScheduler:
        return self._scheduler
