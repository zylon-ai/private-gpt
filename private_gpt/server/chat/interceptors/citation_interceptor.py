from typing import TYPE_CHECKING

from injector import singleton

from private_gpt.components.context.models.context_layer import DocumentLayer
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
from private_gpt.components.engines.citations.utils import (
    extract_sources_from_history,
    process_history_citations,
)

if TYPE_CHECKING:
    from private_gpt.components.engines.citations.types import Document


@singleton
class CitationRequestInterceptor(ChatRequestLoopInterceptor):
    """Populate documents and citations from chat history."""

    async def intercept(self, context: ChatInterceptorContext) -> None:
        """Extract citations and source documents from conversation history."""
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        state = context.state

        documents: list[Document] | None = None
        if state.input.request.citation.enabled:
            (
                chat_history,
                documents,
                potential_citations,
            ) = await process_history_citations(
                state.input.request.messages,
                correlation_id=state.input.request.context.correlation_id,
            )
            state.input.request.messages = chat_history

            if potential_citations:
                existing = state.input.request.citation.citations or []
                additions = [
                    citation
                    for citation in potential_citations
                    if citation not in existing
                ]
                state.input.request.citation.citations = [*existing, *additions]
        else:
            documents = await extract_sources_from_history(state.input.request.messages)

        if documents is not None:
            stack = state.input.context_stack
            stack = stack.remove_layers_of_type(LayerType.DOCUMENT)
            for document in documents:
                stack = stack.append_layer(
                    DocumentLayer(document=document, source="citations")
                )
            state.input.context_stack = stack

        context.set_state(state)
