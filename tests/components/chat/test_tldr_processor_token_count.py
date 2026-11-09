"""condense_chat_history skips the (possibly remote) tokenizer when the history
clearly fits, and still counts exactly near the limit."""

from collections.abc import Sequence
from typing import Any

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.chat.processors.chat_history.memory.strategies.base_strategy import (
    BaseMemoryStrategy,
)
from private_gpt.components.chat.processors.chat_history.memory.tldr_processor import (
    CondenseResponse,
    condense_chat_history,
)
from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizedInput


class _CountingTokenizer:
    """One token per byte: the worst case the byte bound must cover."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, texts: Any = None, images: Any = None, audios: Any = None, **_: Any
    ) -> TokenizedInput:
        self.calls += 1
        text = texts if isinstance(texts, str) else "".join(texts or [])
        return TokenizedInput(list(range(len(text.encode()))))


class _StopStrategy(BaseMemoryStrategy):
    """Stands in for the real strategy; the tests only check whether it is reached."""

    async def get_memory(
        self,
        chat_history: list[ChatMessage],
        max_length: int | None = None,
        **kwargs: Any,
    ) -> list[ChatMessage]:
        raise _Reached


class _Reached(Exception):
    pass


def _history(user_text: str) -> list[ChatMessage]:
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="be brief"),
        ChatMessage(role=MessageRole.USER, content=user_text),
    ]


def _to_prompt(messages: Sequence[ChatMessage]) -> str:
    return "\n".join(m.content or "" for m in messages)


async def _run(
    history: list[ChatMessage], max_length: int
) -> tuple[list[CondenseResponse], int, bool]:
    """Returns the responses, tokenizer calls, and whether condensation was reached."""
    tokenizer = _CountingTokenizer()
    responses: list[CondenseResponse] = []
    try:
        async for r in condense_chat_history(
            history,
            condense_strategy=_StopStrategy(),
            max_length=max_length,
            tokenizer_fn=tokenizer,
            message_to_input=_to_prompt,
        ):
            responses.append(r)
    except _Reached:
        return responses, tokenizer.calls, True
    return responses, tokenizer.calls, False


@pytest.mark.asyncio
async def test_small_history_skips_the_tokenizer() -> None:
    responses, calls, condensing = await _run(_history("hello"), max_length=10_000)

    assert calls == 0
    assert not condensing
    assert len(responses) == 1
    assert not responses[0].is_condensed
    assert [m.content for m in responses[0].chat_history or []] == ["be brief", "hello"]


@pytest.mark.asyncio
async def test_history_near_the_limit_is_counted_exactly() -> None:
    # ~3,000 bytes against a 2,000-token limit: neither bound applies, so the
    # exact count runs (and, at one token per byte here, would need condensing).
    responses, calls, condensing = await _run(
        _history("x" * 1_500 + " " + "y" * 1_500), 2_000
    )

    assert calls >= 1
    assert condensing
    assert responses[0].is_condensed
