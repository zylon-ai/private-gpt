from typing import Any

from injector import singleton

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.context.models.context_layer import ToolDefinitionsLayer
from private_gpt.components.context.models.layer_type import LayerType
from private_gpt.components.engines.chat_loop.interceptors.chat_loop_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_interceptor_context import (
    ChatLoopInterceptorContext,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_phase import (
    InterceptorPhase,
)


@singleton
class NullToolValuesRequestInterceptor(ChatRequestLoopInterceptor):
    """Patch tool specs to strip None kwargs before async invocation."""

    async def intercept(self, context: ChatLoopInterceptorContext) -> None:
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        state = context.state
        tools = [
            self._patch_tool(tool) for tool in state.input.context_stack.all_tools()
        ]

        stack = state.input.context_stack
        stack = stack.remove_layers_of_type(LayerType.TOOL_DEFINITIONS)
        if tools:
            stack = stack.append_layer(
                ToolDefinitionsLayer(tools=tools, source="tool_patch")
            )
        state.input.context_stack = stack
        context.set_state(state)

    def _patch_tool(self, tool: ToolSpec) -> ToolSpec:
        """Patch the tool to avoid to call using optional None params."""

        async def async_patched_tool(*args: Any, **kwargs: Any) -> Any:
            new_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return await tool.async_fn(*args, **new_kwargs)

        tool_copy = tool.model_copy(deep=True)
        tool_copy.async_fn = async_patched_tool
        return tool_copy
