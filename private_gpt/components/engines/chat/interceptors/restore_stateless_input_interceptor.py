from collections import defaultdict, deque

from pydantic import Field

from private_gpt.components.context.models.context_stack import ContextStack
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


class RestoreStatelessInputInterceptorRequest(ChatRequestLoopInterceptor):
    """Revert stateless input fields to their original values before each iteration.

    Interceptors that run later in the before-phase are free to mutate
    ``state.input.request``, ``sampling_params``, and ``llm_kwargs``.
    This interceptor runs first and guarantees those fields always start
    from the original request snapshot, so mutations never bleed from one
    iteration into the next.

    The stateful fields — ``conversation``, ``tools``, and
    ``activated_skills`` — are intentionally left untouched because they
    accumulate across iterations.
    """

    reset_user_instructions: bool = Field(default=True)
    reset_runtime_instructions: bool = Field(default=True)
    reset_documents: bool = Field(default=True)
    reset_tools: bool = Field(default=True)

    def _merge_stack_from_original(
        self, current: ContextStack, original: ContextStack
    ) -> ContextStack:
        reset_types: set[LayerType] = set()
        if self.reset_user_instructions:
            reset_types.add(LayerType.USER_INSTRUCTIONS)
        if self.reset_runtime_instructions:
            reset_types.add(LayerType.RUNTIME_INSTRUCTIONS)
        if self.reset_documents:
            reset_types.add(LayerType.DOCUMENT)
        if self.reset_tools:
            reset_types.add(LayerType.TOOL_DEFINITIONS)

        if not reset_types:
            return current

        original_by_type = defaultdict(deque)
        current_by_type = defaultdict(deque)
        for layer in original.layers:
            original_by_type[layer.type].append(layer)
        for layer in current.layers:
            current_by_type[layer.type].append(layer)

        merged_layers = []
        for original_layer in original.layers:
            layer_type = original_layer.type
            source = original_by_type if layer_type in reset_types else current_by_type
            if source[layer_type]:
                merged_layers.append(source[layer_type].popleft())

        # Keep any extra current layers for non-reset types.
        for layer_type, remaining in current_by_type.items():
            if layer_type in reset_types:
                continue
            merged_layers.extend(remaining)

        return ContextStack(layers=merged_layers)

    async def intercept(self, context: ChatInterceptorContext) -> None:
        """Copy stateless fields from original_input into the current input state."""
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        original = context.state.original_input
        if original is None:
            return

        context.state.input.context_stack = self._merge_stack_from_original(
            current=context.state.input.context_stack,
            original=original.context_stack,
        )

        # Ensure sampling_params and llm_kwargs are always reset
        # to the original values at the start of each iteration.
        context.state.input.sampling_params = dict(original.sampling_params)
        context.state.input.llm_kwargs = dict(original.llm_kwargs)

        context.set_state(context.state)
