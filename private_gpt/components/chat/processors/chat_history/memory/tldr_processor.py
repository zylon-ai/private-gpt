import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from injector import Injector
from llama_index.core.base.llms.types import ChatMessage
from pydantic import BaseModel

from private_gpt.components.chat.processors.chat_history.memory.strategies.base_strategy import (
    BaseMemoryStrategy,
    CondenseStrategyType,
    get_condense_memory_strategy,
)
from private_gpt.components.chat.processors.chat_history.memory.tldr_utils import (
    trim_to_last_tldr,
)
from private_gpt.components.chat.processors.chat_history.memory.utils.condenser import (
    build_condensed_history,
)
from private_gpt.components.chat.processors.chat_history.memory.utils.splitting import (
    get_system_and_conversation_messages,
)
from private_gpt.events.event_errors import Errors
from private_gpt.events.models import (
    BasicContentBlockType,
)
from private_gpt.settings.settings import settings
from private_gpt.utils.tokens import (
    AsyncTokenizerFn,
    MessageInputProtocol,
    estimate_token_count,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if settings().server.debug_mode else logging.INFO)


class CondenseResponse(BaseModel):
    is_condensed: bool = False
    chat_history: list[ChatMessage] | None = None
    condense_blocks: list[BasicContentBlockType] | None = None


CACHE_CONDENSE_STRATEGY: dict[str, BaseMemoryStrategy] = {}


async def condense_chat_history(
    chat_history: list[ChatMessage] | None,
    condense_strategy: BaseMemoryStrategy | None = None,
    strategy_type: str | CondenseStrategyType = CondenseStrategyType.CONDENSER,
    max_length: int | None = None,
    condensation_timeout: int | None = None,
    tokenizer_fn: AsyncTokenizerFn | None = None,
    message_to_input: MessageInputProtocol | None = None,
    injector: Injector | None = None,
    **kwargs: Any,
) -> AsyncGenerator[CondenseResponse]:
    """Condense chat history using specified strategy when length exceeds maximum."""
    system_messages, conversation_history = get_system_and_conversation_messages(
        chat_history or []
    )

    # 1. Remove old messages if TLDR exists
    conversation_history = await asyncio.to_thread(
        trim_to_last_tldr, conversation_history
    )

    # 2. Check if condensation is needed
    if not chat_history or not max_length:
        yield CondenseResponse(chat_history=chat_history, condense_blocks=None)
        return

    # 2. Ensure that max_length - system_messages_length is greater than 0
    remaining_max_length = max_length
    if system_messages:
        system_token_count = await estimate_token_count(
            system_messages,
            tokenizer_fn=tokenizer_fn,
            message_to_input=message_to_input,
            **kwargs,
        )
        if system_token_count >= max_length:
            raise ValueError(
                "System messages alone exceed the maximum length. Cannot condense conversation history."
            )

        remaining_max_length -= system_token_count

    if condense_strategy is None:
        strategy_type = CondenseStrategyType.from_string(str(strategy_type))
        if strategy_type == CondenseStrategyType.UNKNOWN:
            yield CondenseResponse(chat_history=chat_history, condense_blocks=None)
            return

        if str(strategy_type) in CACHE_CONDENSE_STRATEGY:
            condense_strategy = CACHE_CONDENSE_STRATEGY[str(strategy_type)]
        else:
            condense_strategy = get_condense_memory_strategy(
                strategy_type,
                injector=injector,
                message_to_input=message_to_input,
                **kwargs,
            )
            CACHE_CONDENSE_STRATEGY[str(strategy_type)] = condense_strategy

    # 3. Apply condensation if it is needed
    current_token_count = await estimate_token_count(
        conversation_history,
        tokenizer_fn=tokenizer_fn,
        message_to_input=message_to_input,
        **kwargs,
    )
    if condense_strategy is None or current_token_count <= remaining_max_length:
        logger.debug(
            "No condensation needed for conversation history. "
            "Current token count: %d, Available max length: %d, Max length: %d",
            current_token_count,
            remaining_max_length,
            max_length,
        )
        yield CondenseResponse(
            chat_history=system_messages + conversation_history, condense_blocks=None
        )
        return

    yield CondenseResponse(is_condensed=True, chat_history=None, condense_blocks=[])

    logger.debug(
        "Applying condensation strategy '%s' to conversation history. "
        "Current token count: %d, Available max length: %d, Max length: %d",
        strategy_type,
        current_token_count,
        remaining_max_length,
        max_length,
    )
    conversation_history = await _apply_condensation_strategy(
        conversation_history,
        condense_strategy,
        remaining_max_length,
        condensation_timeout,
        **kwargs,
    )

    # 4. Build final history with TLDR
    condensed_history, condense_blocks = await asyncio.to_thread(
        build_condensed_history,
        system_messages=system_messages,
        conversation_history=conversation_history,
    )

    current_token_count = await estimate_token_count(
        condensed_history,
        tokenizer_fn=tokenizer_fn,
        message_to_input=message_to_input,
        **kwargs,
    )
    if current_token_count > max_length:
        raise ValueError(
            "Condensed history exceeds maximum length after applying condensation strategy."
        )

    logger.debug(
        "Applied condensation strategy '%s' to conversation history. "
        "Current token count: %d, Available max length: %d, Max length: %d",
        strategy_type,
        current_token_count,
        remaining_max_length,
        max_length,
    )
    yield CondenseResponse(
        is_condensed=True,
        chat_history=condensed_history,
        condense_blocks=condense_blocks,
    )


async def _apply_condensation_strategy(
    chat_history: list[ChatMessage],
    condense_strategy: BaseMemoryStrategy,
    max_length: int,
    timeout_seconds: int | None = None,
    **kwargs: dict[str, Any],
) -> list[ChatMessage]:
    """Apply the specified condensation strategy to chat history."""
    try:
        coro = condense_strategy.get_memory(
            chat_history=chat_history, max_length=max_length, **kwargs
        )
        if timeout_seconds is None:
            return await coro

        return await asyncio.wait_for(coro, timeout=timeout_seconds)
    except TimeoutError as e:
        raise Errors.Overloaded(
            f"Condensation strategy timed out after {timeout_seconds} seconds",
            event_code=Errors.Codes.OVERLOADED_CONDENSATION_ERROR,
        ) from e
