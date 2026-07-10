from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from injector import inject, singleton

from private_gpt.celery.dispatch import dispatch_task
from private_gpt.components.tools.remote_execution import (
    execute_tool_request,
    invoke_execution_hook,
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
        ToolExecutionResponse,
    )

logger = logging.getLogger(__name__)


class BaseToolScheduler(ABC):
    @property
    def is_async(self) -> bool:
        return False

    @abstractmethod
    async def execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatLoopState | None = None,
        interceptors: list[ToolExecutionInterceptor] | None = None,
    ) -> ToolExecutionResponse:
        ...

    async def async_execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatLoopState | None = None,
        interceptors: list[ToolExecutionInterceptor] | None = None,
    ) -> str:
        del request, state_ctx, interceptors
        raise NotImplementedError

    @abstractmethod
    async def cancel(
        self,
        request: ToolExecutionRequest,
        task_id: str | None = None,
    ) -> bool:
        ...

    async def complete(
        self,
        request: ToolExecutionRequest,
        response: ToolExecutionResponse,
    ) -> None:
        for hook in request.hooks.tool_result:
            await invoke_execution_hook(hook, request, response)


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

    async def cancel(
        self,
        request: ToolExecutionRequest,
        task_id: str | None = None,
    ) -> bool:
        del request, task_id
        return False


@singleton
class CeleryToolScheduler(BaseToolScheduler):
    """Dispatch tool calls to a dedicated Celery tools worker."""

    @inject
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def is_async(self) -> bool:
        return True

    async def execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatLoopState | None = None,
        interceptors: list[ToolExecutionInterceptor] | None = None,
    ) -> ToolExecutionResponse:
        del request, state_ctx, interceptors
        raise NotImplementedError(
            "CeleryToolScheduler only supports async_execute() in resumable chat mode."
        )

    async def cancel(
        self,
        request: ToolExecutionRequest,
        task_id: str | None = None,
    ) -> bool:
        del request
        if not task_id:
            return False

        from private_gpt.celery.celery import celery_app

        celery_app.control.revoke(task_id, terminate=True)
        return True

    async def async_execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatLoopState | None = None,
        interceptors: list[ToolExecutionInterceptor] | None = None,
    ) -> str:
        del state_ctx, interceptors

        correlation_id = request.context.get("correlation_id")
        task_id = f"{correlation_id}:{request.tool_id}" if correlation_id else None
        result = dispatch_task(
            task_name=TOOL_TASK_NAME,
            kwargs={"request_data": request.model_dump(mode="json")},
            queue=self._settings.scheduler.tools.celery_queue,
            task_id=task_id,
        )
        return str(result.id)


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
