import enum
import json
from collections.abc import Callable
from typing import Any, Literal

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.bridge.pydantic import Field, model_validator
from llama_index.core.llms.llm import LLM
from llama_index.core.memory.types import (
    DEFAULT_CHAT_STORE_KEY,
    BaseChatStoreMemory,
)
from llama_index.core.storage.chat_store import BaseChatStore, SimpleChatStore

from private_gpt.components.llm.llm_helper import TokenizerFn
from private_gpt.utils.tokens import async_tokenizer

DEFAULT_TOKEN_LIMIT_RATIO = 0.75
DEFAULT_TOKEN_LIMIT = 3000


class TrimStrategy(enum.StrEnum):
    """Strategy for trimming messages."""

    FIRST = "first"  # Keep first messages up to token limit
    LAST = "last"  # Keep last messages up to token limit


def _default_text_splitter(text: str) -> list[str]:
    """Default text splitter that splits on newlines."""
    splits = text.split("\n")
    return [s + "\n" for s in splits[:-1]] + splits[-1:]


def _is_message_type(
    message: ChatMessage, message_types: MessageRole | list[MessageRole]
) -> bool:
    """Check if message matches one of the specified types."""
    if isinstance(message_types, MessageRole):
        message_types = [message_types]
    return message.role in message_types


class TrimmingMemory(BaseChatStoreMemory):
    """Advanced buffer for storing and managing chat history with trimming capabilities.

    This memory buffer extends the basic ChatMemoryBuffer with advanced features:
    - Message trimming with configurable strategies
    - System message preservation
    - Partial message support
    - Flexible token counting
    """

    token_limit: int
    trim_strategy: TrimStrategy = TrimStrategy.LAST
    include_system: bool = True
    allow_partial: bool = False
    start_on: MessageRole | list[MessageRole] | None = None
    end_on: MessageRole | list[MessageRole] | None = None
    text_splitter: Callable[[str], list[str]] = Field(
        default_factory=lambda: _default_text_splitter,
        exclude=True,
    )
    tokenizer_fn: TokenizerFn = Field(
        exclude=True,
    )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "TrimmingMemory"

    @model_validator(mode="before")
    @classmethod
    def validate_memory(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Validate memory configuration."""
        # Validate token limit
        token_limit = values.get("token_limit", -1)
        if token_limit < 1:
            raise ValueError("Token limit must be set and greater than 0.")

        # Validate tokenizer
        tokenizer_fn = values.get("tokenizer_fn")
        if tokenizer_fn is None:
            # TODO: Replace with a default tokenizer function
            raise ValueError("tokenizer_fn must be provided.")

        # Validate text splitter
        text_splitter = values.get("text_splitter")
        if text_splitter is None:
            values["text_splitter"] = _default_text_splitter

        # Validate strategy-specific constraints
        trim_strategy = values.get("trim_strategy", TrimStrategy.LAST)
        start_on = values.get("start_on")
        include_system = values.get("include_system", True)

        if start_on and trim_strategy == TrimStrategy.FIRST:
            raise ValueError("start_on can only be used with 'last' strategy")

        if include_system and trim_strategy == TrimStrategy.FIRST:
            raise ValueError("include_system can only be used with 'last' strategy")

        return values

    @classmethod
    def from_defaults(
        cls,
        chat_history: list[ChatMessage] | None = None,
        llm: LLM | None = None,
        chat_store: BaseChatStore | None = None,
        chat_store_key: str = DEFAULT_CHAT_STORE_KEY,
        token_limit: int | None = None,
        trim_strategy: TrimStrategy = TrimStrategy.LAST,
        include_system: bool = True,
        allow_partial: bool = False,
        start_on: MessageRole | list[MessageRole] | None = None,
        end_on: MessageRole | list[MessageRole] | None = None,
        tokenizer_fn: TokenizerFn | None = None,
        text_splitter: Callable[[str], list[str]] | None = None,
        **kwargs: Any,
    ) -> "TrimmingMemory":
        """Create an advanced chat memory buffer from an LLM."""
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")

        if llm is not None:
            context_window = llm.metadata.context_window
            token_limit = token_limit or int(context_window * DEFAULT_TOKEN_LIMIT_RATIO)
        elif token_limit is None:
            token_limit = DEFAULT_TOKEN_LIMIT

        if chat_history is not None:
            chat_store = chat_store or SimpleChatStore()
            chat_store.set_messages(chat_store_key, chat_history)

        return cls(
            token_limit=token_limit,
            trim_strategy=trim_strategy,
            include_system=include_system,
            allow_partial=allow_partial,
            start_on=start_on,
            end_on=end_on,
            tokenizer_fn=tokenizer_fn,  # type: ignore
            text_splitter=text_splitter or _default_text_splitter,
            chat_store=chat_store or SimpleChatStore(),
            chat_store_key=chat_store_key,
        )

    async def aget(
        self, input: str | None = None, initial_token_count: int = 0, **kwargs: Any
    ) -> list[ChatMessage]:
        """Get trimmed chat history based on configured strategy."""
        chat_history = await self.aget_all()

        if not chat_history:
            return []

        if initial_token_count > self.token_limit:
            raise ValueError("Initial token count exceeds token limit")

        max_tokens = self.token_limit - initial_token_count

        if self.trim_strategy == TrimStrategy.FIRST:
            return await self._trim_first_max_tokens(chat_history, max_tokens)
        else:
            return await self._trim_last_max_tokens(chat_history, max_tokens)

    async def _trim_first_max_tokens(
        self, messages: list[ChatMessage], max_tokens: int
    ) -> list[ChatMessage]:
        """Keep the first messages up to the token limit."""
        if not messages:
            return messages

        # Find the maximum number of messages we can include
        idx = 0
        for i in range(len(messages)):
            current_messages = messages[: len(messages) - i] if i else messages
            if await self._token_count_for_messages(current_messages) <= max_tokens:
                idx = len(messages) - i
                break

        # Handle partial messages if allowed
        if self.allow_partial and idx < len(messages):
            idx = await self._try_include_partial_message(
                messages, idx, max_tokens, "first"
            )

        # Apply end_on constraint
        if self.end_on:
            while idx > 0 and not _is_message_type(messages[idx - 1], self.end_on):
                idx -= 1

        return messages[:idx]

    async def _trim_last_max_tokens(
        self, messages: list[ChatMessage], max_tokens: int
    ) -> list[ChatMessage]:
        """Keep the last messages up to the token limit."""
        if not messages:
            return []

        # Handle end_on constraint first
        working_messages = messages[:]
        if self.end_on:
            while working_messages and not _is_message_type(
                working_messages[-1], self.end_on
            ):
                working_messages.pop()

        # Check if we need to preserve system message
        has_system = (
            self.include_system
            and working_messages
            and working_messages[0].role == MessageRole.SYSTEM
        )

        if has_system:
            system_msg = working_messages[0]
            remaining_messages = working_messages[1:]
            system_tokens = await self._token_count_for_messages([system_msg])
            available_tokens = max_tokens - system_tokens

            if available_tokens <= 0:
                return []

            # Trim the non-system messages
            trimmed_remaining = await self._trim_messages_from_end(
                remaining_messages, available_tokens
            )

            result = [system_msg, *trimmed_remaining]
        else:
            result = await self._trim_messages_from_end(working_messages, max_tokens)

        # Apply start_on constraint
        if self.start_on and result:
            start_idx = 0
            system_offset = 1 if has_system else 0

            # Find first occurrence of start_on message type
            for i in range(system_offset, len(result)):
                if _is_message_type(result[i], self.start_on):
                    start_idx = i
                    break

            if has_system and start_idx > 0:
                result = [result[0], *result[start_idx:]]
            elif not has_system:
                result = result[start_idx:]

        return result

    async def _trim_messages_from_end(
        self, messages: list[ChatMessage], max_tokens: int
    ) -> list[ChatMessage]:
        """Trim messages from the end to fit within token limit."""
        if not messages:
            return messages

        # Start from the end and work backwards
        for i in range(len(messages)):
            current_messages = messages[-(len(messages) - i) :] if i else messages
            if await self._token_count_for_messages(current_messages) <= max_tokens:
                idx = len(messages) - i
                break
        else:
            idx = 0

        # Handle partial messages if allowed
        if self.allow_partial and idx > 0:
            idx = await self._try_include_partial_message(
                messages, idx - 1, max_tokens, "last"
            )

        # Ensure we don't start with assistant or tool messages
        final_messages = messages[-(len(messages) - idx) :] if idx else []
        while final_messages and final_messages[0].role in (
            MessageRole.ASSISTANT,
            MessageRole.TOOL,
        ):
            final_messages = final_messages[1:]

        return final_messages

    async def _try_include_partial_message(
        self,
        messages: list[ChatMessage],
        idx: int,
        max_tokens: int,
        direction: Literal["first", "last"],
    ) -> int:
        """Try to include a partial message if allow_partial is True."""
        if idx >= len(messages):
            return idx

        excluded = messages[idx].model_copy(deep=True)

        # Only handle string content for now
        if isinstance(excluded.content, str):
            text = excluded.content
            split_texts = self.text_splitter(text)

            if direction == "last":
                split_texts = list(reversed(split_texts))

            # Try progressively smaller portions
            for i in range(1, len(split_texts)):
                partial_texts = split_texts[:-i]
                partial_content = "".join(partial_texts)

                if direction == "last":
                    partial_content = "".join(reversed(partial_texts))

                excluded.content = partial_content

                test_messages = (
                    [*messages[:idx], excluded]
                    if direction == "first"
                    else [*messages[idx:], excluded]
                )

                if await self._token_count_for_messages(test_messages) <= max_tokens:
                    messages[idx] = excluded
                    return idx + 1

        return idx

    async def _token_count_for_messages(self, messages: list[ChatMessage]) -> int:
        """Count tokens for a list of messages."""
        if not messages:
            return 0

        # Convert messages to string representation for token counting
        msg_str = " ".join(str(m.content) for m in messages)
        return len(await async_tokenizer(msg_str, tokenizer_fn=self.tokenizer_fn))

    def to_string(self) -> str:
        """Convert memory to string."""
        return self.json()

    @classmethod
    def from_string(cls, json_str: str) -> "TrimmingMemory":
        """Create memory from string."""
        dict_obj = json.loads(json_str)
        return cls.from_dict(dict_obj)

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Convert memory to dict."""
        return self.dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any], **kwargs: Any) -> "TrimmingMemory":
        """Create memory from dict."""
        from llama_index.core.storage.chat_store.loading import load_chat_store

        # Handle backwards compatibility
        if "chat_history" in data:
            chat_history = data.pop("chat_history")
            simple_store = SimpleChatStore(store={DEFAULT_CHAT_STORE_KEY: chat_history})
            data["chat_store"] = simple_store
        elif "chat_store" in data:
            chat_store_dict = data.pop("chat_store")
            chat_store = load_chat_store(chat_store_dict)
            data["chat_store"] = chat_store

        return cls(**data)
