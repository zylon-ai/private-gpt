import asyncio
from typing import Any

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ChatRequest
from private_gpt.components.context.models.context_layer import (
    ContextPromptLayer,
    UserInstructionsLayer,
)
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
from private_gpt.components.engines.chat_loop.utils.request_builder import (
    build_request_from_context_stack,
)
from private_gpt.components.llm.llm_helper import as_sync_tokenizer_fn
from private_gpt.components.prompts.prompt_builder import PromptBuilderService


@singleton
class SystemPromptRequestInterceptor(ChatRequestLoopInterceptor):
    """Ensure final system prompt is present in conversation history."""

    @inject
    def __init__(
        self,
        prompt_builder_service: PromptBuilderService,
        add_context_to_system_prompt: bool | None = None,
    ) -> None:
        self._prompt_builder_service = prompt_builder_service
        self._add_context_to_system_prompt = add_context_to_system_prompt

    async def intercept(self, context: ChatLoopInterceptorContext) -> None:
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        state = context.state.model_copy(deep=True)
        stack = state.input.context_stack
        stack = stack.remove_layers_of_source("platform_header")
        stack = stack.remove_layers_of_type(LayerType.CONTEXT)

        generated_layers = await self._build_generated_layers(
            context=context,
            request=state.input.request,
            documents=stack.all_documents(),
        )
        for layer in generated_layers:
            stack = stack.append_layer(layer)

        state.input.context_stack = stack
        state.input.request = build_request_from_context_stack(
            state.input.request, state.input.context_stack
        )
        context.set_state(state)

    async def _build_generated_layers(
        self,
        context: ChatLoopInterceptorContext,
        request: ChatRequest,
        documents: list[Any],
    ) -> list[UserInstructionsLayer | ContextPromptLayer]:
        def build() -> list[UserInstructionsLayer | ContextPromptLayer]:
            generated_layers: list[UserInstructionsLayer | ContextPromptLayer] = []
            header = self._prompt_builder_service.create_chat_header_prompt().format()
            if header.strip():
                generated_layers.append(
                    UserInstructionsLayer(
                        text=header.strip(),
                        source="platform_header",
                    )
                )

            add_context_to_system_prompt = (
                self._add_context_to_system_prompt
                if self._add_context_to_system_prompt is not None
                else request.context.add_context_to_system_prompt
            )
            if add_context_to_system_prompt:
                (
                    context_prompt,
                    _,
                ) = self._prompt_builder_service.create_chat_context_for_system_prompt(
                    documents=documents,
                    generate_citations=request.citation.enabled,
                    guidelines_prompt=None,
                    token_limit=context.state.runtime.effective_token_limit,
                    tokenizer_fn=as_sync_tokenizer_fn(
                        context.state.runtime.tokenizer_fn
                    ),
                )
                if context_prompt:
                    context_text = context_prompt.format().strip()
                    if context_text:
                        generated_layers.append(
                            ContextPromptLayer(
                                text=context_text,
                                source="system_prompt",
                            )
                        )

            return generated_layers

        return await asyncio.to_thread(build)
