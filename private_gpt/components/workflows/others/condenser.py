import logging
from typing import Any

from llama_index.core.base.llms.generic_utils import messages_to_history_str
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.schema import QueryType
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from pydantic import Field

from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.di import get_global_injector

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Maximum number of tokens allowed for the condense operation
_MAX_CONDENSE_TOKENS = 30  # Around 21 words (30 * 0.7)


class CondenseInputEvent(StartEvent):
    """Event to start the condense workflow."""

    query: QueryType = Field(..., description="The user query to condense.")
    max_condense_tokens: int = Field(
        default=_MAX_CONDENSE_TOKENS,
        description="The maximum number of tokens for the condense operation.",
    )
    chat_history: list[ChatMessage] = Field(
        default_factory=list, description="The chat history."
    )


class CondenseResultEvent(StopEvent):
    """Event indicating condense workflow completion."""

    condensed_query: QueryType = Field(..., description="The condensed query.")
    original_query: QueryType = Field(..., description="The original query.")


class CondenserWorkflow(Workflow):
    """Condenses a user query with chat history into a standalone question."""

    def __init__(
        self,
        llm: LLM,
        prompt_builder_service: PromptBuilderService | None = None,
        callback_manager: CallbackManager | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ):
        """Initialize the CondenserWorkflow."""
        super().__init__(timeout=timeout)
        self._llm = llm
        self.prompt_builder_service = (
            prompt_builder_service or get_global_injector().get(PromptBuilderService)
        )

        # Set the callback manager for the LLM
        if callback_manager:
            self._llm.callback_manager = callback_manager

    @step
    async def condense_question(self, ev: CondenseInputEvent) -> CondenseResultEvent:
        """Condense a user query with chat history into a standalone question."""
        # Get query and chat history from context
        query: QueryType = ev.query
        chat_history: list[ChatMessage] = ev.chat_history
        max_condense_tokens: int = ev.max_condense_tokens

        # Skip condensing if chat history is empty
        if ev.chat_history is None or len(ev.chat_history) == 0:
            logger.debug("Skipping question condensing")
            return CondenseResultEvent(
                condensed_query=ev.query, original_query=ev.query
            )

        try:
            token_limit = self._llm.metadata.context_window - max_condense_tokens
            if token_limit < 0:
                return CondenseResultEvent(condensed_query=query, original_query=query)

            memory = ChatMemoryBuffer.from_defaults(
                chat_history=chat_history,
                llm=self._llm,
                token_limit=token_limit,
            )

            # Format chat history as a string
            chat_history_str = messages_to_history_str(memory.get())
            max_words = int(max_condense_tokens * 0.7)
            condense_prompt_builder = (
                self.prompt_builder_service.create_chat_condense_prompt(
                    question=str(query),
                    chat_history=chat_history_str,
                    max_words=max_words,
                )
            )
            logger.debug(f"Chat history for condensing: {chat_history_str}")

            # Get condensed question from LLM
            condensed_question = await self._llm.acomplete(
                condense_prompt_builder.format(), max_tokens=max_condense_tokens
            )

            final_condensed_query = str(condensed_question).strip()
            logger.debug(f"Condensed query: {final_condensed_query}")

            return CondenseResultEvent(
                condensed_query=final_condensed_query, original_query=query
            )
        except Exception as e:
            logger.error(f"Error in condense_question: {e}")
            return CondenseResultEvent(condensed_query=query, original_query=query)
