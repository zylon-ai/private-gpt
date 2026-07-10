from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from private_gpt.components.llm.llm_helper import get_tokenizer_fn
from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizedInput, TokenizerBase
from private_gpt.utils.tokens import async_tokenizer, estimate_token_count


class AsyncCapableTokenizer(MagicMock):
    def __init__(self):
        super().__init__(spec=TokenizerBase)
        self.sync_calls = 0
        self.async_calls = 0

    def __call__(self, texts=None, images=None, audios=None, **kwargs):
        del images, audios, kwargs
        self.sync_calls += 1
        return TokenizedInput(input_ids=[99])

    async def acall(self, texts=None, images=None, audios=None, **kwargs):
        del images, audios, kwargs
        self.async_calls += 1
        text = texts or ""
        return TokenizedInput(input_ids=list(range(len(str(text).split()))))


@pytest.mark.asyncio
async def test_async_tokenizer_prefers_underlying_acall():
    tokenizer = AsyncCapableTokenizer()
    tokenizer_fn = get_tokenizer_fn(tokenizer)

    tokens = await async_tokenizer("one two three", tokenizer_fn=tokenizer_fn)

    assert tokens == [0, 1, 2]
    assert tokenizer.async_calls == 1
    assert tokenizer.sync_calls == 0


@pytest.mark.asyncio
async def test_estimate_token_count_uses_async_tokenizer_wrapper():
    tokenizer = AsyncCapableTokenizer()
    tokenizer_fn = get_tokenizer_fn(tokenizer)

    async def message_to_input(messages, **kwargs):
        del kwargs
        raise AssertionError("message_to_input should run synchronously in this test")

    count = await estimate_token_count(
        chat_history=[ChatMessage(role=MessageRole.USER, content="one two three")],
        tokenizer_fn=tokenizer_fn,
        message_to_input=lambda messages: str(messages[0].content),
    )

    assert count == 3
    assert tokenizer.async_calls == 1
    assert tokenizer.sync_calls == 0
