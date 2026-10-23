from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import LLM
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.workflow import Context

from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.components.workflows.others.condenser import (
    CondenserWorkflow,
)
from private_gpt.components.workflows.retrieval.retrieval import (
    RetrieverWorkflow,
)
from private_gpt.components.workflows.retrieval.semantic_search import (
    SemanticSearchInputEvent,
    SemanticSearchResultEvent,
    SemanticSearchWorkflow,
)


@pytest.fixture
def mock_llm() -> AsyncMock:
    mock = AsyncMock(spec=LLM)
    mock.acomplete.return_value = "Condensed query"
    return mock


@pytest.fixture
def mock_retriever() -> AsyncMock:
    mock = AsyncMock(spec=BaseRetriever)
    nodes = [
        NodeWithScore(node=TextNode(text="Test content 1", id_="node1"), score=0.9),
        NodeWithScore(node=TextNode(text="Test content 2", id_="node2"), score=0.8),
    ]
    mock.aretrieve.return_value = nodes
    return mock


@pytest.fixture
def mock_prompt_builder() -> AsyncMock:
    mock = AsyncMock(spec=PromptBuilderService)
    mock.create_chat_condense_prompt.return_value = MagicMock()
    mock.create_chat_condense_prompt.return_value.format.return_value = (
        "formatted prompt"
    )
    return mock


@pytest.fixture
def mock_condenser_workflow(mock_llm: LLM) -> CondenserWorkflow:
    workflow = CondenserWorkflow(
        llm=mock_llm,
    )
    return workflow


@pytest.fixture
def mock_retriever_workflow(
    mock_llm: LLM, mock_retriever: BaseRetriever
) -> RetrieverWorkflow:
    workflow = RetrieverWorkflow(
        retriever=mock_retriever,
    )
    return workflow


@pytest.fixture
def chat_history() -> list[ChatMessage]:
    return [
        ChatMessage(role=MessageRole.USER, content="Hello, how are you?"),
        ChatMessage(
            role=MessageRole.ASSISTANT, content="I'm doing well, how can I help you?"
        ),
        ChatMessage(
            role=MessageRole.USER, content="I want to know about the meaning of life"
        ),
    ]


@pytest.mark.asyncio
async def test_semantic_search_workflow_init() -> None:
    mock_llm = AsyncMock(spec=LLM)
    mock_retriever = AsyncMock(spec=BaseRetriever)

    workflow = SemanticSearchWorkflow(
        llm=mock_llm,
        retriever=mock_retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )

    assert workflow._condenser_workflow is not None
    assert workflow._retriever_workflow is not None

    mock_condenser = AsyncMock(spec=CondenserWorkflow)
    mock_retriever_wf = AsyncMock(spec=RetrieverWorkflow)

    workflow = SemanticSearchWorkflow(
        llm=mock_llm,
        retriever=mock_retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
        condenser_workflow=mock_condenser,
        retriever_workflow=mock_retriever_wf,
    )

    assert workflow._condenser_workflow is mock_condenser
    assert workflow._retriever_workflow is mock_retriever_wf


@pytest.mark.asyncio
async def test_start_step_no_condensing(
    mock_llm: AsyncMock, mock_retriever: AsyncMock, chat_history: list[ChatMessage]
) -> None:
    workflow = SemanticSearchWorkflow(
        llm=mock_llm,
        retriever=mock_retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )

    ctx = Context(workflow=workflow)
    input_event = SemanticSearchInputEvent(
        query="test query",
        use_condense=False,
        token_limit=100,
        chat_history=chat_history,
        kwargs={"similarity_cutoff": 0.5},
    )
    output = await workflow.run(ctx, start_event=input_event)

    assert isinstance(output, SemanticSearchResultEvent)
    assert output.retrieval is not None
    assert output.condense is None


@pytest.mark.asyncio
async def test_start_step_with_condensing(
    mock_llm: AsyncMock, mock_retriever: AsyncMock, chat_history: list[ChatMessage]
) -> None:
    workflow = SemanticSearchWorkflow(
        llm=mock_llm,
        retriever=mock_retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)],
    )

    ctx = Context(workflow=workflow)
    input_event = SemanticSearchInputEvent(
        query="test query",
        use_condense=True,
        token_limit=100,
        chat_history=chat_history,
        kwargs={"similarity_cutoff": 0.5},
    )
    output = await workflow.run(ctx, start_event=input_event)

    assert isinstance(output, SemanticSearchResultEvent)
    assert output.retrieval is not None
    assert output.condense is not None
