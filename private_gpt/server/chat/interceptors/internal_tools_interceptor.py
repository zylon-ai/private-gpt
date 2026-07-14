from injector import inject, singleton

from private_gpt.components.context.models.context_layer import (
    ToolDefinitionsLayer,
)
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
from private_gpt.components.engines.chat.utils.request_builder import (
    build_request_from_context_stack,
)
from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.components.tools.tool_pipeline import ToolPipeline


@singleton
class InternalToolRequestInterceptor(ChatRequestLoopInterceptor):
    @inject
    def __init__(
        self,
        tool_pipeline: ToolPipeline,
        prompt_builder: PromptBuilderService,
    ) -> None:
        self._tool_pipeline = tool_pipeline
        self._prompt_builder = prompt_builder

    async def intercept(self, context: ChatInterceptorContext) -> None:
        if (
            context.phase != InterceptorPhase.VALIDATION
            and context.phase != InterceptorPhase.BEFORE_ITERATION
        ):
            return

        state = context.state
        stack = state.input.context_stack

        tool_request = build_request_from_context_stack(state.input.request, stack)
        final_tool_request = await self._tool_pipeline.contextualize_internal_tools(
            tool_request
        )
        tools = final_tool_request.tool_config.tools

        if context.phase == InterceptorPhase.BEFORE_ITERATION and tools:
            tools = self._prompt_builder.seed_tool_instructions(tools)

        stack = state.input.context_stack
        stack = stack.remove_layers_of_type(LayerType.TOOL_DEFINITIONS)
        if tools:
            stack = stack.append_layer(
                ToolDefinitionsLayer(tools=tools, source="internal_tools")
            )
        state.input.context_stack = stack
        context.set_state(state)
