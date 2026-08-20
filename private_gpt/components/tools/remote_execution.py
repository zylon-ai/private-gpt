from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, cast

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import adapt_to_async_tool
from pydantic import BaseModel, Field, model_validator

from private_gpt.components.chat.models.chat_config_models import (
    ToolExecutionMetadata,
    ToolSpec,
)
from private_gpt.components.engines.chat.models.chat_phase import (
    InterceptorPhase,
)
from private_gpt.components.engines.chat.models.execution_hooks import (
    ExecutionHooks,
)
from private_gpt.components.engines.chat.utils.tool_utils import execute_tool_call
from private_gpt.components.tools.tool_execution_outcome import (
    ToolExecutionError,
    ToolExecutionFailure,
    ToolExecutionOutcome,
    ToolExecutionSuccess,
)
from private_gpt.context import snapshot
from private_gpt.events.models import (
    NO_TOOL_CONTENT,
    TextBlock,
    from_tool_output,
    normalize_tool_result_content,
)

if TYPE_CHECKING:
    from llama_index.core.tools import AsyncBaseTool

    from private_gpt.components.engines.chat.models.chat_state import (
        ChatState,
    )
    from private_gpt.components.engines.chat.models.execution_hooks import (
        ToolExecutionHook,
    )


class ToolExecutionRequest(BaseModel):
    tool_id: str
    tool_name: str
    tool_kwargs: dict[str, Any] = Field(default_factory=dict)
    tool_spec: ToolSpec
    context: dict[str, Any] = Field(default_factory=dict)
    hooks: ExecutionHooks = Field(default_factory=ExecutionHooks)
    interceptor_paths: list[str] = Field(default_factory=list)


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
    outcome: ToolExecutionOutcome
    tool_message: ChatMessage

    @model_validator(mode="before")
    @classmethod
    def upgrade_legacy_outcome(cls, value: Any) -> Any:
        if not isinstance(value, dict) or "outcome" in value:
            return value
        upgraded = dict(value)
        content = upgraded.pop("result_content", [])
        is_error = upgraded.pop("is_error", False)
        upgraded["outcome"] = (
            {
                "type": "failure",
                "error": {
                    "message": _result_content_text(content),
                    "details": {"content": content},
                },
            }
            if is_error
            else {"type": "success", "content": content}
        )
        return upgraded

    @property
    def result_content(self) -> list[Any]:
        if isinstance(self.outcome, ToolExecutionSuccess):
            return self.outcome.content
        details = self.outcome.error.details.get("content", [])
        return cast(list[Any], details) if isinstance(details, list) else []

    @property
    def is_error(self) -> bool:
        return isinstance(self.outcome, ToolExecutionFailure)


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


def tool_execution_interceptor_paths(
    interceptors: list[ToolExecutionInterceptor] | None,
) -> list[str]:
    return [
        f"{type(interceptor).__module__}:{type(interceptor).__qualname__}"
        for interceptor in interceptors or []
    ]


def resolve_tool_execution_interceptors(
    paths: list[str],
) -> list[ToolExecutionInterceptor]:
    from private_gpt.di import get_global_injector

    injector = get_global_injector(True)
    return [injector.get(_import_callable(path)) for path in paths]


class ToolExecutor:
    def __init__(
        self,
        interceptors: list[ToolExecutionInterceptor] | None = None,
    ) -> None:
        self._interceptors = interceptors or []

    async def execute(
        self,
        request: ToolExecutionRequest,
        state_ctx: ChatState | None = None,
    ) -> ToolExecutionResponse:
        tool_kwargs = dict(request.tool_kwargs)
        try:
            tool = await rebuild_tool_from_spec(request.tool_spec)

            before_context = ToolExecutionInterceptorContext(
                phase=InterceptorPhase.BEFORE_TOOL,
                request=request,
                tool_kwargs=tool_kwargs,
            )
            for interceptor in self._interceptors:
                await interceptor.intercept(before_context)
            tool_kwargs = before_context.tool_kwargs

            result, tool_message = await execute_tool_call(
                tool=tool,
                tool_name=request.tool_name,
                tool_id=request.tool_id,
                tool_kwargs=tool_kwargs,
                state_ctx=state_ctx,
            )
            result_content = normalize_tool_result_content(
                from_tool_output(result.tool_output.raw_output)
                if result.tool_output.raw_output is not None
                else [TextBlock(text=result.tool_output.content or NO_TOOL_CONTENT)]
            )
            outcome: ToolExecutionOutcome = (
                ToolExecutionFailure(
                    error=ToolExecutionError(
                        message=result.tool_output.content
                        or _result_content_text(result_content),
                        details={"content": result_content},
                    )
                )
                if result.tool_output.is_error
                else ToolExecutionSuccess(content=result_content)
            )
            response = ToolExecutionResponse(
                tool_name=request.tool_name,
                tool_id=request.tool_id,
                outcome=outcome,
                tool_message=tool_message,
            )

            after_context = ToolExecutionInterceptorContext(
                phase=InterceptorPhase.AFTER_TOOL,
                request=request,
                tool_kwargs=tool_kwargs,
                response=response,
            )
            for interceptor in self._interceptors:
                await interceptor.intercept(after_context)

            assert after_context.response is not None
            return after_context.response
        except Exception as exc:
            message = str(exc)
            return ToolExecutionResponse(
                tool_name=request.tool_name,
                tool_id=request.tool_id,
                outcome=ToolExecutionFailure(
                    error=ToolExecutionError(
                        message=message,
                        exception_type=type(exc).__name__,
                        details={"content": [TextBlock(text=message)]},
                    )
                ),
                tool_message=ChatMessage(
                    role="tool",
                    content=message,
                    additional_kwargs={
                        "tool_call_id": request.tool_id,
                        "tool_call_name": request.tool_name,
                        "tool_call_args": tool_kwargs,
                        "raw_output": message,
                    },
                ),
            )


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
        return adapt_to_async_tool(tool_spec.to_function_tool())

    rebuilt = await _invoke_rebuild(metadata)
    return adapt_to_async_tool(rebuilt.to_function_tool())


async def execute_tool_request(
    request: ToolExecutionRequest,
    state_ctx: ChatState | None = None,
    interceptors: list[ToolExecutionInterceptor] | None = None,
) -> ToolExecutionResponse:
    executor = ToolExecutor(interceptors=interceptors)
    return await executor.execute(request, state_ctx=state_ctx)


def build_tool_execution_context(state: ChatState) -> dict[str, Any]:
    correlation_id = state.input.request.context.correlation_id
    return {
        "correlation_id": correlation_id,
        "message_id": correlation_id,
        "messages": [
            msg.model_dump(mode="json", exclude_none=True)
            for msg in state.input.request.messages
        ],
        # ContextVars don't cross the broker boundary; carry the request's
        # context bag so the Celery tools worker can reinstall it around tool
        # execution (see tool_run_task).
        "_context": snapshot(),
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


def _result_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content or "Tool execution failed")
