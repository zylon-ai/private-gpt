from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import adapt_to_async_tool
from pydantic import BaseModel, Field

from private_gpt.components.chat.models.chat_config_models import (
    ToolExecutionMetadata,
    ToolSpec,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_phase import (
    InterceptorPhase,
)
from private_gpt.components.engines.chat_loop.models.execution_hooks import (
    ExecutionHooks,
)
from private_gpt.components.engines.chat_loop.utils.tool_utils import execute_tool_call
from private_gpt.events.models import (
    ResultContentBlockType,
    TextBlock,
    from_tool_output,
)

if TYPE_CHECKING:
    from llama_index.core.tools import AsyncBaseTool

    from private_gpt.components.engines.chat_loop.models.chat_loop_state import (
        ChatLoopState,
    )
    from private_gpt.components.engines.chat_loop.models.execution_hooks import (
        ToolExecutionHook,
    )


class ToolExecutionRequest(BaseModel):
    tool_id: str
    tool_name: str
    tool_kwargs: dict[str, Any] = Field(default_factory=dict)
    tool_spec: ToolSpec
    context: dict[str, Any] = Field(default_factory=dict)
    hooks: ExecutionHooks = Field(default_factory=ExecutionHooks)


async def invoke_execution_hook(
    hook: ToolExecutionHook,
    request: ToolExecutionRequest,
    response: ToolExecutionResponse,
) -> None:
    callback_callable = _import_callable(hook.callable_path)
    result = callback_callable(request=request, response=response, **hook.kwargs)
    if inspect.isawaitable(result):
        await result


class ToolExecutionResponse(BaseModel):
    tool_name: str
    tool_id: str
    result_content: list[ResultContentBlockType] = Field(default_factory=list)
    is_error: bool = False
    tool_message: ChatMessage


class ToolExecutionInterceptorContext(BaseModel):
    phase: InterceptorPhase
    request: ToolExecutionRequest
    tool_kwargs: dict[str, Any]
    response: ToolExecutionResponse | None = None

    def set_tool_kwargs(self, tool_kwargs: dict[str, Any]) -> None:
        self.tool_kwargs = tool_kwargs

    def set_response(self, response: ToolExecutionResponse) -> None:
        self.response = response


class ToolExecutionInterceptor(ABC):
    @abstractmethod
    async def intercept(self, context: ToolExecutionInterceptorContext) -> None:
        """Mutate tool execution context before/after tool invocation."""


class ToolExecutor:
    def __init__(
        self,
        interceptors: list[ToolExecutionInterceptor] | None = None,
    ) -> None:
        self._interceptors = interceptors or []

    async def execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatLoopState | None = None,
    ) -> ToolExecutionResponse:
        tool = await rebuild_tool_from_spec(request.tool_spec)

        before_context = ToolExecutionInterceptorContext(
            phase=InterceptorPhase.BEFORE_TOOL,
            request=request,
            tool_kwargs=dict(request.tool_kwargs),
        )
        for interceptor in self._interceptors:
            await interceptor.intercept(before_context)

        result, tool_message = await execute_tool_call(
            tool=tool,
            tool_name=request.tool_name,
            tool_id=request.tool_id,
            tool_kwargs=before_context.tool_kwargs,
            state_ctx=state_ctx,
        )
        response = ToolExecutionResponse(
            tool_name=request.tool_name,
            tool_id=request.tool_id,
            result_content=(
                from_tool_output(result.tool_output.raw_output)
                if result.tool_output.raw_output is not None
                else [TextBlock(text=result.tool_output.content or "")]
            ),
            is_error=result.tool_output.is_error,
            tool_message=tool_message,
        )

        after_context = ToolExecutionInterceptorContext(
            phase=InterceptorPhase.AFTER_TOOL,
            request=request,
            tool_kwargs=before_context.tool_kwargs,
            response=response,
        )
        for interceptor in self._interceptors:
            await interceptor.intercept(after_context)

        assert after_context.response is not None
        return after_context.response


def build_rebuild_metadata(
    rebuild_callable: Any,
    rebuild_kwargs: dict[str, Any] | None = None,
) -> ToolExecutionMetadata:
    return ToolExecutionMetadata(
        rebuild_callable=_callable_path(rebuild_callable),
        rebuild_kwargs=rebuild_kwargs or {},
    )


async def rebuild_tool_from_spec(tool_spec: ToolSpec) -> AsyncBaseTool:
    metadata = tool_spec.execution_metadata
    if metadata is None:
        raise ValueError(f"Tool '{tool_spec.name}' is missing execution metadata.")

    rebuilt = await _invoke_rebuild(metadata)
    return adapt_to_async_tool(rebuilt.to_function_tool())


async def execute_tool_request(
    request: ToolExecutionRequest,
    state_ctx: ChatLoopState | None = None,
    interceptors: list[ToolExecutionInterceptor] | None = None,
) -> ToolExecutionResponse:
    executor = ToolExecutor(interceptors=interceptors)
    return await executor.execute(request, state_ctx=state_ctx)


def build_tool_execution_context(state: ChatLoopState) -> dict[str, Any]:
    return {
        "correlation_id": state.input.request.context.correlation_id,
        "messages": [
            msg.model_dump(mode="json", exclude_none=True)
            for msg in state.input.request.messages
        ],
    }


def restore_chat_history_from_context(context: dict[str, Any]) -> list[ChatMessage]:
    return [
        ChatMessage.model_validate(message_data)
        for message_data in context.get("messages", [])
    ]


async def _invoke_rebuild(metadata: ToolExecutionMetadata) -> ToolSpec:
    rebuild_callable = _import_callable(metadata.rebuild_callable)
    rebuilt = rebuild_callable(**metadata.rebuild_kwargs)
    if inspect.isawaitable(rebuilt):
        rebuilt = await rebuilt
    if not isinstance(rebuilt, ToolSpec):
        raise TypeError("Tool rebuild callable must return a ToolSpec instance.")
    return rebuilt


def _callable_path(rebuild_callable: Any) -> str:
    return f"{rebuild_callable.__module__}:{rebuild_callable.__qualname__}"


def _import_callable(path: str) -> Any:
    module_name, attr_path = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    target = module
    for attr in attr_path.split("."):
        target = getattr(target, attr)
    return target
