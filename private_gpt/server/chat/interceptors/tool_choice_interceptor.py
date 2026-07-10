from injector import singleton
from llama_index.core.base.llms.types import MessageRole

from private_gpt.components.chat.processors.chat_history.tools.tool_choices import (
    process_tool_choices,
)
from private_gpt.components.context.models.context_layer import ToolDefinitionsLayer
from private_gpt.components.context.models.layer_type import LayerType
from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import (
    InterceptorPhase,
)


@singleton
class ToolChoiceRequestInterceptor(ChatRequestLoopInterceptor):
    """Filter tools and update user message hint for forced tool choice."""

    async def intercept(self, context: ChatInterceptorContext) -> None:
        """Apply tool choice policy to history and available tools."""
        if context.phase == InterceptorPhase.BEFORE_ITERATION:
            await self.intercept_before(context)
        elif context.phase == InterceptorPhase.AFTER_ITERATION:
            await self.intercept_after(context)

    async def intercept_before(self, context: ChatInterceptorContext) -> None:
        state = context.state
        history, tools, new_tool_choice = await process_tool_choices(
            chat_history=state.input.request.messages,
            tools=state.input.context_stack.all_tools(),
            tool_choices=state.input.request.tool_config.tool_choices,
        )
        new_state = state.model_copy(deep=True)
        new_state.input.request.messages = list(history or [])
        if new_tool_choice:
            new_state.input.request.tool_config.tool_choices = new_tool_choice

        stack = new_state.input.context_stack
        stack = stack.remove_layers_of_type(LayerType.TOOL_DEFINITIONS)
        if tools:
            stack = stack.append_layer(
                ToolDefinitionsLayer(tools=list(tools), source="tool_choice")
            )
        new_state.input.context_stack = stack

        context.set_state(new_state)

    async def intercept_after(self, context: ChatInterceptorContext) -> None:
        state = context.state
        tool_choices = state.input.request.tool_config.tool_choices

        if not tool_choices or tool_choices in ("auto", "none"):
            return

        if not state.original_input:
            return

        new_state = state.model_copy(deep=True)

        original_messages = state.original_input.request.messages
        original_last_user = next(
            (
                msg
                for msg in reversed(original_messages)
                if msg.role == MessageRole.USER
            ),
            None,
        )
        if original_last_user:
            messages = new_state.input.request.messages
            last_user_idx = next(
                (
                    i
                    for i in range(len(messages) - 1, -1, -1)
                    if messages[i].role == MessageRole.USER
                ),
                None,
            )
            if last_user_idx is not None:
                messages[last_user_idx] = original_last_user.model_copy(deep=True)

        original_tools = state.original_input.context_stack.all_tools()
        stack = new_state.input.context_stack.remove_layers_of_type(
            LayerType.TOOL_DEFINITIONS
        )
        if original_tools:
            stack = stack.append_layer(
                ToolDefinitionsLayer(tools=original_tools, source="tool_choice")
            )
        new_state.input.context_stack = stack
        new_state.input.request.tool_config = (
            new_state.input.request.tool_config.model_copy(
                update={"tool_choices": "auto"}
            )
        )

        context.set_state(new_state)
