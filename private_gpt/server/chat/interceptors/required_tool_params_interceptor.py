from __future__ import annotations

from typing import TYPE_CHECKING

from injector import singleton

from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_phase import (
    InterceptorPhase,
)
from private_gpt.components.tools.remote_execution import (
    ToolExecutionInterceptor,
    ToolExecutionInterceptorContext,
)
from private_gpt.components.tools.utils import require_tool_params

if TYPE_CHECKING:
    from private_gpt.components.engines.chat.models.chat_interceptor_context import (
        ChatInterceptorContext,
    )


@singleton
class RequiredToolParamsInterceptor(
    ChatRequestLoopInterceptor,
    ToolExecutionInterceptor,
):
    """Reject tool calls that omit a schema-required parameter.

    Runs after ``NullToolValuesRequestInterceptor`` so a required field sent as
    ``null`` is treated as missing rather than accepted.
    """

    async def intercept(
        self,
        context: ChatInterceptorContext | ToolExecutionInterceptorContext,
    ) -> None:
        if not isinstance(context, ToolExecutionInterceptorContext):
            return
        if context.phase != InterceptorPhase.BEFORE_TOOL:
            return

        require_tool_params(
            context.request.tool_name,
            context.tool_kwargs,
            context.request.tool_spec.input_schema,
        )
