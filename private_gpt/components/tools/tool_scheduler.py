from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from injector import inject, singleton

from private_gpt.celery.dispatch import dispatch_task
from private_gpt.components.streaming.stream_component import StreamComponent
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

    @abstractmethod
    async def cancel(
        self,
        request: ToolExecutionRequest,
        task_id: str | None = None,
    ) -> bool:
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
    def __init__(self, settings: Settings, stream_component: StreamComponent) -> None:
        self._settings = settings
        self._stream_service = stream_component.stream

    async def execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatLoopState | None = None,
        interceptors: list[ToolExecutionInterceptor] | None = None,
    ) -> ToolExecutionResponse:
        del state_ctx, interceptors

        correlation_id = request.context.get("correlation_id")
        task_id = None
        if correlation_id:
            task_id = f"{correlation_id}:{request.tool_id}"

        result = dispatch_task(
            task_name=TOOL_TASK_NAME,
            kwargs={"request_data": request.model_dump(mode="json")},
            queue=self._settings.scheduler.tools.celery_queue,
            task_id=task_id,
        )
        cancelled = False

        try:
            while not result.ready():
                if correlation_id and await self._stream_service.is_cancelled(
                    correlation_id
                ):
                    await self.cancel(request, result.id)
                    cancelled = True
                    raise asyncio.CancelledError(
                        f"Tool execution cancelled for correlation_id={correlation_id}"
                    )
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            if not cancelled:
                await self.cancel(request, result.id)
            raise

        if result.failed():
            error_content = f"Tool execution failed in worker: {str(result.result)}"
            from private_gpt.events.models import TextBlock
            from llama_index.core.base.llms.types import ChatMessage
            
            return ToolExecutionResponse(
                tool_name=request.tool_name,
                tool_id=request.tool_id,
                result_content=[TextBlock(text=error_content)],
                is_error=True,
                tool_message=ChatMessage(
                    role="tool",
                    content=error_content,
                    additional_kwargs={
                        "tool_call_id": request.tool_id,
                        "tool_call_name": request.tool_name,
                        "tool_call_args": request.tool_kwargs,
                        "raw_output": error_content,
                    },
                ),
            )

        return ToolExecutionResponse.model_validate(result.result)

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
