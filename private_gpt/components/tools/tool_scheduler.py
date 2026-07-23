from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from asyncio import CancelledError, to_thread
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from celery.exceptions import TimeoutError as CeleryTimeoutError
from injector import Injector, inject, singleton

from private_gpt.celery.dispatch import dispatch_task
from private_gpt.components.tools.remote_execution import (
    execute_tool_request,
    invoke_execution_hook,
    tool_execution_interceptor_paths,
)
from private_gpt.settings.settings import Settings, settings

TOOL_TASK_NAME = "private_gpt.tools.run"

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from private_gpt.components.engines.chat.models.chat_state import ChatState
    from private_gpt.components.tools.remote_execution import (
        ToolExecutionInterceptor,
        ToolExecutionRequest,
        ToolExecutionResponse,
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if settings().server.debug_mode else logging.INFO)


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
    ) -> ToolExecutionResponse: ...

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
    ) -> bool: ...

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
        try:
            return await execute_tool_request(
                request, state_ctx=state_ctx, interceptors=interceptors
            )
        except Exception:
            logger.exception("Local tool '%s' execution failed", request.tool_name)
            raise

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
        self._async_cancel = True
        self._background_tasks: set[asyncio.Task[Any]] = set()

    @property
    def is_async(self) -> bool:
        return True

    async def execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatState | None = None,
        interceptors: list[ToolExecutionInterceptor] | None = None,
    ) -> ToolExecutionResponse:
        del state_ctx
        request = request.model_copy(
            update={"interceptor_paths": tool_execution_interceptor_paths(interceptors)}
        )
        correlation_id = request.context.get("correlation_id")
        message_id = request.context.get("message_id") or correlation_id
        logger.debug(
            "Dispatching blocking tool execution correlation_id=%s "
            "message_id=%s tool_id=%s tool_name=%s queue=%s",
            correlation_id,
            message_id,
            request.tool_id,
            request.tool_name,
            self._settings.scheduler.tools.celery_queue,
        )
        result = dispatch_task(
            task_name=TOOL_TASK_NAME,
            kwargs={"request_data": request.model_dump(mode="json")},
            queue=self._settings.scheduler.tools.celery_queue,
            ignore_result=False,
        )
        logger.debug(
            "Blocking tool execution dispatched correlation_id=%s "
            "message_id=%s task_id=%s tool_id=%s tool_name=%s queue=%s",
            correlation_id,
            message_id,
            result.id,
            request.tool_id,
            request.tool_name,
            self._settings.scheduler.tools.celery_queue,
        )
        try:
            response_data = await to_thread(
                result.get,
                timeout=self._settings.scheduler.tools.callback_timeout_seconds,
            )
        except (CancelledError, CeleryTimeoutError):
            await self.cancel_task(str(result.id))
            raise

        from private_gpt.components.tools.remote_execution import ToolExecutionResponse

        return ToolExecutionResponse.model_validate(response_data)

    async def cancel(
        self,
        request: ToolExecutionRequest,
        task_id: str | None = None,
    ) -> bool:
        del request
        return await self.cancel_task(task_id) if task_id else False

    def _spawn(self, coro: Coroutine[Any, Any, Any], *, name: str) -> None:
        task = asyncio.create_task(coro, name=name)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def cancel_task(self, task_id: str) -> bool:
        return await (
            self._cancel_task_async(task_id)
            if self._async_cancel
            else asyncio.to_thread(self._cancel_task_sync, task_id)
        )

    async def _cancel_task_async(self, task_id: str) -> bool:
        coro = asyncio.to_thread(self._cancel_task_sync, task_id)
        self._spawn(coro, name=f"cancel_tool_task_{task_id}")
        return True

    def _cancel_task_sync(self, task_id: str) -> bool:
        from private_gpt.celery.celery import celery_app

        logger.info(
            "Tool cancellation started task_id=%s",
            task_id,
        )
        try:
            celery_app.control.revoke(task_id, terminate=True)
        except Exception:
            logger.exception("Tool cancellation failed task_id=%s", task_id)
            raise
        logger.info(
            "Tool cancellation finished task_id=%s",
            task_id,
        )
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
        message_id = request.context.get("message_id") or correlation_id
        task_id = f"{correlation_id}:{request.tool_id}" if correlation_id else None
        logger.debug(
            "Dispatching async tool execution correlation_id=%s "
            "message_id=%s task_id=%s tool_id=%s tool_name=%s queue=%s",
            correlation_id,
            message_id,
            task_id,
            request.tool_id,
            request.tool_name,
            self._settings.scheduler.tools.celery_queue,
        )
        result = dispatch_task(
            task_name=TOOL_TASK_NAME,
            kwargs={"request_data": request.model_dump(mode="json")},
            queue=self._settings.scheduler.tools.celery_queue,
            task_id=task_id,
            ignore_result=True,
        )
        logger.debug(
            "Async tool execution dispatched correlation_id=%s "
            "message_id=%s task_id=%s tool_id=%s tool_name=%s queue=%s",
            correlation_id,
            message_id,
            result.id,
            request.tool_id,
            request.tool_name,
            self._settings.scheduler.tools.celery_queue,
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
