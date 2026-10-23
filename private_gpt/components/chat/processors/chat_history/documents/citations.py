from typing import Any

from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from llama_index.core.base.llms.types import (
    TextBlock as LITextBlock,
)

from private_gpt.components.engines.citations.types import Document
from private_gpt.components.engines.citations.utils import (
    deduplicate_documents_in_history,
    extract_sources_from_history,
)


async def process_chat_history_with_documents(
    chat_history: list[ChatMessage] | None,
    documents: list[Document] | None = None,
    deduplicate_documents: bool = True,
    force_to_return_citations: bool = False,
    add_context_to_system_prompt: bool = False,
    **kwargs: dict[str, Any],
) -> list[ChatMessage] | None:
    if not documents or not chat_history:
        return chat_history

    # Force to return citations
    if force_to_return_citations:
        last_user_message = next(
            (msg for msg in reversed(chat_history) if msg.role == MessageRole.USER),
            None,
        )

        suffix = "Use citations to back up your answer."
        if (
            last_user_message
            and last_user_message.content
            and suffix not in last_user_message.content
        ):
            last_user_message.blocks = [
                LITextBlock(text=f"{last_user_message.content}. {suffix}")
            ]

    if deduplicate_documents and chat_history:
        # Deduplicate documents in the chat history to avoid sending the
        # same document multiple times to the LLM, keeping only the last
        # occurrence of each document so that the most recent version and
        # context of each document is preserved.
        chat_history = await deduplicate_documents_in_history(chat_history)

    if add_context_to_system_prompt and chat_history:
        # We need to remove the context from the chat history
        # to avoid sending it to the LLM twice
        for i in range(len(chat_history)):
            has_sources = any(await extract_sources_from_history([chat_history[i]]))
            if has_sources:
                chat_history[i].blocks = [
                    LITextBlock(
                        text="Information retrieved successfully. Context has been updated."
                    )
                ]

    return chat_history
