from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from injector import inject, singleton

from private_gpt.components.tools.remote_execution import (
    ToolExecutionResponse,
    execute_tool_request,
)
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.engines.chat_loop.models.chat_loop_state import (
        ChatLoopState,
    )
    from private_gpt.components.tools.remote_execution import ToolExecutionRequest

logger = logging.getLogger(__name__)


class BaseToolScheduler(ABC):
    @abstractmethod
    async def execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatLoopState | None = None,
    ) -> ToolExecutionResponse:
        ...


@singleton
class LocalToolScheduler(BaseToolScheduler):
    """Execute tools in-process (no worker dispatch)."""

    async def execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatLoopState | None = None,
    ) -> ToolExecutionResponse:
        return await execute_tool_request(request, state_ctx=state_ctx)


@singleton
class CeleryToolScheduler(BaseToolScheduler):
    """Dispatch tool calls to a dedicated Celery tools worker."""

    @inject
    def __init__(self, settings: Settings) -> None:
        from private_gpt.celery.celery import celery_app

        self._celery_app = celery_app
        self._tools_queue = settings.scheduler.tools.celery_queue

    async def execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatLoopState | None = None,
    ) -> ToolExecutionResponse:
        del state_ctx

        result = self._celery_app.send_task(
            "private_gpt.tools.run",
            kwargs=request.model_dump(mode="json"),
            queue=self._tools_queue,
        )

        while not result.ready():
            await asyncio.sleep(0.1)

        if result.failed():
            raise result.result

        return ToolExecutionResponse.model_validate(result.result)


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
