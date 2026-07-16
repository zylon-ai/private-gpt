from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING

from injector import Injector, inject, singleton

from private_gpt.celery.dispatch import dispatch_task
from private_gpt.components.tools.remote_execution import (
    execute_tool_request,
    invoke_execution_hook,
    tool_execution_interceptor_paths,
)
from private_gpt.settings.settings import Settings

TOOL_TASK_NAME = "private_gpt.tools.run"

if TYPE_CHECKING:
    from private_gpt.components.engines.chat.models.chat_state import ChatState
    from private_gpt.components.tools.remote_execution import (
        ToolExecutionInterceptor,
        ToolExecutionRequest,
        ToolExecutionResponse,
    )

logger = logging.getLogger(__name__)

ToolSchedulerProvider = (
    type["BaseToolScheduler"] | Callable[[Injector], "BaseToolScheduler"]
)
_TOOL_SCHEDULERS: dict[str, ToolSchedulerProvider] = {}


def register_tool_scheduler(mode: str, provider: ToolSchedulerProvider) -> None:
    _TOOL_SCHEDULERS[mode] = provider


class BaseToolScheduler(ABC):
    @property
    def is_async(self) -> bool:
        return False

    @abstractmethod
    async def execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatState | None = None,
        interceptors: list[ToolExecutionInterceptor] | None = None,
    ) -> ToolExecutionResponse:
        ...

    async def async_execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatState | None = None,
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

    async def cancel_task(self, task_id: str) -> bool:
        del task_id
        return False

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
        state_ctx: ChatState | None = None,
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

    async def cancel_task(self, task_id: str) -> bool:
        del task_id
        return False


register_tool_scheduler("local", LocalToolScheduler)


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
        state_ctx: ChatState | None = None,
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
        return await self.cancel_task(task_id) if task_id else False

    async def cancel_task(self, task_id: str) -> bool:
        from private_gpt.celery.celery import celery_app

        celery_app.control.revoke(task_id, terminate=True)
        return True

    async def async_execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatState | None = None,
        interceptors: list[ToolExecutionInterceptor] | None = None,
    ) -> str:
        del state_ctx

        request = request.model_copy(
            update={"interceptor_paths": tool_execution_interceptor_paths(interceptors)}
        )

        correlation_id = request.context.get("correlation_id")
        task_id = f"{correlation_id}:{request.tool_id}" if correlation_id else None
        result = dispatch_task(
            task_name=TOOL_TASK_NAME,
            kwargs={"request_data": request.model_dump(mode="json")},
            queue=self._settings.scheduler.tools.celery_queue,
            task_id=task_id,
        )
        return str(result.id)


register_tool_scheduler("celery", CeleryToolScheduler)


@singleton
class ToolSchedulerFactory:
    @inject
    def __init__(self, settings: Settings, injector: Injector) -> None:
        self._settings = settings
        self._injector = injector
        self._scheduler: BaseToolScheduler | None = None

    def get(self) -> BaseToolScheduler:
        if self._scheduler is None:
            mode = self._settings.scheduler.tools.mode
            provider = _TOOL_SCHEDULERS.get(mode)
            if provider is None:
                raise ValueError(f"Unknown scheduler.tools.mode: {mode}")
            self._scheduler = (
                self._injector.get(provider)
                if isinstance(provider, type)
                else provider(self._injector)
            )
        return self._scheduler
