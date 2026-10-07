"""Summary LLM calls carry an explicit max_tokens.

Without it the Triton LLM generates up to "context window - prompt". A condenser
summary on test3 then ran for minutes and starved the Triton backend, stalling
every other stream until zylon-gpt's 60 s stall guard fired.
"""

from typing import Any

import pytest
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from private_gpt.components.workflows.others.summary import (
    SUMMARY_MAX_OUTPUT_TOKENS,
)
from private_gpt.components.workflows.others.tree_summarize_synthesizer import (
    TreeSummarizeSynthesizer,
)


class RecordingLLM(CustomLLM):
    chat_model: bool = True
    calls: list[dict[str, Any]] = Field(default_factory=list)
    reply: str = "summary"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192, num_output=256, is_chat_model=self.chat_model
        )

    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        self.calls.append(kwargs)
        return CompletionResponse(text=self.reply)

    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        raise NotImplementedError

    async def acomplete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        return self.complete(prompt, formatted=formatted, **kwargs)

    def chat(self, messages: Any, **kwargs: Any) -> ChatResponse:
        self.calls.append(kwargs)
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=self.reply)
        )

    async def achat(self, messages: Any, **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)


class Summary(BaseModel):
    text: str


TEMPLATE = PromptTemplate("Summarize {query_str}:\n{context_str}")


def _synth(llm: RecordingLLM, **kwargs: Any) -> TreeSummarizeSynthesizer:
    return TreeSummarizeSynthesizer(
        llm=llm,
        summary_template=TEMPLATE,
        llm_kwargs={"max_tokens": 123},
        **kwargs,
    )


@pytest.mark.parametrize("chat_model", [True, False])
@pytest.mark.parametrize("n_chunks", [1, 3])
async def test_async_summary_calls_pass_max_tokens(
    chat_model: bool, n_chunks: int
) -> None:
    llm = RecordingLLM(chat_model=chat_model, calls=[])
    synth = _synth(llm)
    # Chunks too big to repack into one: a tree of calls (leaves, then root).
    chunks = ["word " * 2500 for _ in range(n_chunks)]
    out = await synth.aget_response(query_str="the logs", text_chunks=chunks)
    assert out == "summary"
    assert llm.calls
    assert all(call.get("max_tokens") == 123 for call in llm.calls)


@pytest.mark.parametrize("chat_model", [True, False])
def test_sync_summary_calls_pass_max_tokens(chat_model: bool) -> None:
    llm = RecordingLLM(chat_model=chat_model, calls=[])
    synth = _synth(llm)
    out = synth.get_response(query_str="q", text_chunks=["a", "b"])
    assert out == "summary"
    assert llm.calls
    assert all(c.get("max_tokens") == 123 for c in llm.calls)


async def test_structured_summary_calls_pass_max_tokens() -> None:
    llm = RecordingLLM(calls=[], reply='{"text": "s"}')
    synth = _synth(llm, output_cls=Summary)
    out = await synth.aget_response(query_str="q", text_chunks=["a"])
    assert out == Summary(text="s")
    assert llm.calls
    assert all(c.get("max_tokens") == 123 for c in llm.calls)


def test_summary_budget_is_a_few_thousand_tokens() -> None:
    assert 1024 <= SUMMARY_MAX_OUTPUT_TOKENS <= 8192
