from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from injector import inject, singleton

from private_gpt.celery.dispatch import dispatch_task
from private_gpt.components.tools.remote_execution import (
    ToolExecutionResponse,
    execute_tool_request,
)
from private_gpt.settings.settings import Settings

TOOL_TASK_NAME = "private_gpt.tools.run"

if TYPE_CHECKING:
    from private_gpt.components.engines.chat_loop.models.chat_loop_state import (
        ChatLoopState,
    )
    from private_gpt.components.tools.remote_execution import (
        ToolExecutionInterceptor,
        ToolExecutionRequest,
    )

logger = logging.getLogger(__name__)


class BaseToolScheduler(ABC):
    @abstractmethod
    async def execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatLoopState | None = None,
        interceptors: list[ToolExecutionInterceptor] | None = None,
    ) -> ToolExecutionResponse:
        ...


@singleton
class LocalToolScheduler(BaseToolScheduler):
    """Execute tools in-process (no worker dispatch)."""

    async def execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatLoopState | None = None,
        interceptors: list[ToolExecutionInterceptor] | None = None,
    ) -> ToolExecutionResponse:
        return await execute_tool_request(
            request, state_ctx=state_ctx, interceptors=interceptors
        )


@singleton
class CeleryToolScheduler(BaseToolScheduler):
    """Dispatch tool calls to a dedicated Celery tools worker."""

    @inject
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatLoopState | None = None,
        interceptors: list[ToolExecutionInterceptor] | None = None,
    ) -> ToolExecutionResponse:
        del state_ctx, interceptors

        result = dispatch_task(
            task_name=TOOL_TASK_NAME,
            args=(request,),
            queue=self._settings.scheduler.tools.celery_queue,
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
