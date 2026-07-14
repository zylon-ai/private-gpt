from injector import inject, singleton

from private_gpt.components.chat.processors.chat_history.documents.citations import (
    process_chat_history_with_documents,
)
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
class DocumentProcessingRequestInterceptor(ChatRequestLoopInterceptor):
    """Apply document-context formatting to chat history."""

    _add_context_to_system_prompt: bool | None = None

    @inject
    def __init__(self, add_context_to_system_prompt: bool | None = None) -> None:
        self._add_context_to_system_prompt = add_context_to_system_prompt

    async def intercept(self, context: ChatInterceptorContext) -> None:
        """Render document context into conversation history when configured."""
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        state = context.state
        history = await process_chat_history_with_documents(
            chat_history=state.input.request.messages,
            documents=state.input.context_stack.all_documents(),
            force_to_return_citations=(
                state.input.request.citation.enabled
                and state.input.request.citation.force_to_return_citations
            ),
            add_context_to_system_prompt=(
                self._add_context_to_system_prompt
                if self._add_context_to_system_prompt is not None
                else state.input.request.context.add_context_to_system_prompt
            ),
            deduplicate_documents=state.input.request.context.deduplicate_context_in_history,
        )
        new_state = state.model_copy(deep=True)
        new_state.input.request.messages = history or []
        context.set_state(new_state)
