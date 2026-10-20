import asyncio
from typing import Any
from unittest.mock import MagicMock

from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.chat.processors.chat_history.memory.strategies.condenser import (
    CondenserContextMemoryStrategy,
)


def test_summary_left_sends_each_batch_only_its_own_messages() -> None:
    strategy = CondenserContextMemoryStrategy(
        summarize_workflow_builder=MagicMock(),
        prompt_builder_service=MagicMock(),
        llm_component=MagicMock(),
    )
    calls: list[list[str]] = []

    async def fake_summarize_obj(**kwargs: Any) -> None:
        calls.append(kwargs["texts"])
        return None

    async def fake_tokens(messages: list[ChatMessage], **_: Any) -> int:
        return 10 * len(messages)

    strategy._summarize_obj = fake_summarize_obj  # type: ignore[method-assign]
    strategy._get_messages_tokens = fake_tokens  # type: ignore[method-assign]

    history = [
        ChatMessage(
            role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"message-{i}",
        )
        for i in range(6)
    ]

    asyncio.run(
        strategy._summary_left(
            llm=MagicMock(),
            tokenizer_fn=None,
            chat_history=history,
            max_length=50,
            chunk_size=2,
        )
    )

    assert len(calls) > 1
    assert all(len(texts) == 1 for texts in calls)
    joined = [texts[0] for texts in calls]
    assert len(set(joined)) == len(joined)
