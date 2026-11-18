import copy

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock

from private_gpt.components.chunk.models import Chunk
from private_gpt.components.engines.citations.types import Document
from private_gpt.components.engines.citations.utils import (
    deduplicate_documents_in_history,
    extract_sources_from_history,
)
from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.components.prompts.prompt_template import PromptTemplateService
from private_gpt.events.models import SourceBlock
from private_gpt.server.ingest.model import IngestedDoc


@pytest.fixture
def prompt_builder() -> PromptBuilderService:
    return PromptBuilderService(PromptTemplateService())


def _chunk(cid: str) -> Chunk:
    return Chunk(
        object="context.chunk",
        id=cid,
        score=0.8,
        document=IngestedDoc(artifact="doc", doc_metadata={"file_name": "doc.pdf"}),
        text=f"content of {cid}",
    )


def _doc_id(cid: str) -> str:
    return Document.from_source(_chunk(cid)).id_


def _tool_result(pb: PromptBuilderService, chunks: list[Chunk]) -> ChatMessage:
    prompt, _ = pb.create_context_prompt(
        documents=[Document.from_source(c) for c in chunks], generate_citations=True
    )
    return ChatMessage(
        role=MessageRole.TOOL,
        blocks=[TextBlock(text=prompt.format())],
        additional_kwargs={"source": [SourceBlock(sources=chunks)]},
    )


def _history(pb: PromptBuilderService) -> list[ChatMessage]:
    # Turn 3 re-retrieves chunk "a" from turn 1.
    a, b, c, d = (_chunk(x) for x in "abcd")
    return [
        ChatMessage(role=MessageRole.USER, content="q1"),
        _tool_result(pb, [a, b]),
        ChatMessage(role=MessageRole.ASSISTANT, content="a1"),
        ChatMessage(role=MessageRole.USER, content="q2"),
        _tool_result(pb, [c]),
        ChatMessage(role=MessageRole.ASSISTANT, content="a2"),
        ChatMessage(role=MessageRole.USER, content="q3"),
        _tool_result(pb, [a, d]),
    ]


async def _doc_ids(msg: ChatMessage) -> list[str]:
    return [doc.id_ for doc in await extract_sources_from_history([msg])]


async def test_keep_first_is_the_default(prompt_builder: PromptBuilderService) -> None:
    explicit = await deduplicate_documents_in_history(
        _history(prompt_builder), prompt_builder, keep="first"
    )
    default = await deduplicate_documents_in_history(
        _history(prompt_builder), prompt_builder
    )
    assert explicit is not None
    assert default is not None
    assert [m.content for m in explicit] == [m.content for m in default]


async def test_keep_first_leaves_earlier_messages_unchanged(
    prompt_builder: PromptBuilderService,
) -> None:
    history = _history(prompt_builder)
    before = copy.deepcopy(history)

    result = await deduplicate_documents_in_history(
        history, prompt_builder, keep="first"
    )

    assert result is not None
    assert [m.content for m in result[:7]] == [m.content for m in before[:7]]
    assert _doc_id("a") in await _doc_ids(result[1])
    assert await _doc_ids(result[7]) == [_doc_id("d")]


async def test_keep_first_replaces_fully_duplicated_message(
    prompt_builder: PromptBuilderService,
) -> None:
    a = _chunk("a")
    history = [
        ChatMessage(role=MessageRole.USER, content="q1"),
        _tool_result(prompt_builder, [a]),
        ChatMessage(role=MessageRole.ASSISTANT, content="a1"),
        ChatMessage(role=MessageRole.USER, content="q2"),
        _tool_result(prompt_builder, [a]),
    ]

    result = await deduplicate_documents_in_history(
        history, prompt_builder, keep="first"
    )

    assert result is not None
    assert await _doc_ids(result[4]) == []
    assert "earlier message" in (result[4].content or "")
    assert await _doc_ids(result[1]) == [_doc_id("a")]


async def test_keep_last_moves_repeated_document_to_newest_message(
    prompt_builder: PromptBuilderService,
) -> None:
    history = _history(prompt_builder)
    original_first = history[1].content

    result = await deduplicate_documents_in_history(
        history, prompt_builder, keep="last"
    )

    assert result is not None
    assert _doc_id("a") not in await _doc_ids(result[1])
    assert _doc_id("a") in await _doc_ids(result[7])
    assert result[1].content != original_first
