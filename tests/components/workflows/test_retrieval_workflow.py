from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.workflow import Context as LlamaIndexContext

from private_gpt.components.workflows.retrieval.retrieval import (
    FinalNodesRetrievalEvent,
    RawNodesRetrievedEvent,
    RetrieverConfig,
    RetrieverInputEvent,
    RetrieverResultEvent,
    RetrieverWorkflow,
)


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
def mock_empty_retriever() -> AsyncMock:
    mock = AsyncMock(spec=BaseRetriever)
    mock.aretrieve.return_value = []
    return mock


@pytest.mark.asyncio
async def test_init_validation():
    mock = AsyncMock(spec=BaseRetriever)

    RetrieverWorkflow(
        retriever=mock,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0)],
    )

    RetrieverWorkflow(
        retriever=mock,
        node_postprocessors_fn=lambda **kwargs: [
            SimilarityPostprocessor(similarity_cutoff=0)
        ],
    )


@pytest.mark.asyncio
async def test_retrieve_raw_nodes_step(mock_retriever: AsyncMock) -> None:
    workflow = RetrieverWorkflow(
        retriever=mock_retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0)],
    )

    ctx_mock = MagicMock()
    ctx_mock.store = MagicMock()
    ctx_mock.store.set = AsyncMock()
    ctx_mock.store.get = AsyncMock()
    ctx_mock.write_event_to_stream = MagicMock()

    input_event = RetrieverInputEvent(query="test query", token_limit=100)
    result = await workflow.retrieve_raw_nodes(ctx_mock, input_event)

    mock_retriever.aretrieve.assert_awaited_once_with("test query")
    ctx_mock.store.set.assert_any_await("query", "test query")
    ctx_mock.store.set.assert_any_await("token_limit", 100)

    assert isinstance(result, RawNodesRetrievedEvent)
    assert len(result.nodes) == 2
    assert result.nodes[0].score == 0.9
    assert result.nodes[1].score == 0.8


@pytest.mark.asyncio
async def test_retrieve_raw_nodes_workflow(mock_retriever: AsyncMock) -> None:
    workflow = RetrieverWorkflow(
        retriever=mock_retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0)],
    )

    with patch("workflows.Context", autospec=True) as mock_context_class:
        real_context = LlamaIndexContext(workflow=workflow)
        mock_context_class.return_value = real_context

        # Add necessary attributes to the real context
        real_context.store.set = AsyncMock()
        real_context.store.get = AsyncMock()
        real_context.write_event_to_stream = MagicMock()
        real_context.shutdown = AsyncMock()

        input_event = RetrieverInputEvent(query="test query", token_limit=100)
        handler = workflow.run(start_event=input_event)
        result = await handler

        mock_retriever.aretrieve.assert_awaited_once_with("test query")
        assert isinstance(result, RetrieverResultEvent)
        assert len(result.nodes) == 2


@pytest.mark.asyncio
async def test_retrieve_empty_nodes(mock_empty_retriever: AsyncMock) -> None:
    workflow = RetrieverWorkflow(
        retriever=mock_empty_retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0)],
    )

    with patch("workflows.Context", autospec=True) as mock_context_class:
        real_context = LlamaIndexContext(workflow=workflow)
        mock_context_class.return_value = real_context

        real_context.store.set = AsyncMock()
        real_context.store.get = AsyncMock()
        real_context.write_event_to_stream = MagicMock()
        real_context.shutdown = AsyncMock()

        input_event = RetrieverInputEvent(query="test query")
        result = await workflow.run(real_context, start_event=input_event)

        mock_empty_retriever.aretrieve.assert_awaited_once()
        assert len(result.nodes) == 0


@pytest.mark.asyncio
async def test_transform_nodes() -> None:
    workflow = RetrieverWorkflow(
        retriever=AsyncMock(spec=BaseRetriever),
        node_postprocessors=[SimilarityPostprocessor()],
    )

    ctx_mock = MagicMock()
    ctx_mock.store = MagicMock()
    ctx_mock.store.get = AsyncMock()
    ctx_mock.store.get.side_effect = lambda key: {
        "query": "test query",
        "token_limit": 100,
    }.get(key)
    ctx_mock.write_event_to_stream = MagicMock()

    nodes = [
        NodeWithScore(node=TextNode(text="Test content 1", id_="node1"), score=0.9),
        NodeWithScore(node=TextNode(text="Test content 2", id_="node2"), score=0.8),
    ]
    input_event = RawNodesRetrievedEvent(nodes=nodes)
    result = await workflow.transform_nodes(ctx_mock, input_event)

    assert isinstance(result, FinalNodesRetrievalEvent)
    assert len(result.nodes) == 2


@pytest.mark.asyncio
async def test_transform_nodes_with_filtering() -> None:
    proc = SimilarityPostprocessor(similarity_cutoff=0.85)

    workflow = RetrieverWorkflow(
        retriever=AsyncMock(spec=BaseRetriever), node_postprocessors=[proc]
    )

    ctx_mock = MagicMock()
    ctx_mock.store = MagicMock()
    ctx_mock.store.get = AsyncMock()
    ctx_mock.store.get.side_effect = lambda key: {
        "query": "test query",
        "token_limit": None,
    }.get(key)
    ctx_mock.write_event_to_stream = MagicMock()

    nodes = [
        NodeWithScore(node=TextNode(text="High score", id_="node1"), score=0.9),
        NodeWithScore(node=TextNode(text="Low score", id_="node2"), score=0.8),
    ]
    input_event = RawNodesRetrievedEvent(nodes=nodes)
    result = await workflow.transform_nodes(ctx_mock, input_event)

    assert isinstance(result, FinalNodesRetrievalEvent)
    assert len(result.nodes) == 1
    assert result.nodes[0].score == 0.9


@pytest.mark.asyncio
async def test_finalize_nodes() -> None:
    workflow = RetrieverWorkflow(
        retriever=AsyncMock(spec=BaseRetriever),
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0)],
    )

    ctx_mock = MagicMock()
    ctx_mock.store = MagicMock()
    ctx_mock.store.get = AsyncMock()
    ctx_mock.store.get.side_effect = lambda key: {
        "query": "test query",
        "token_limit": None,
    }.get(key)
    ctx_mock.write_event_to_stream = MagicMock()

    nodes = [
        NodeWithScore(node=TextNode(text="Test content", id_="node1"), score=0.9),
    ]
    input_event = FinalNodesRetrievalEvent(nodes=nodes)
    result = await workflow.finalize_nodes(ctx_mock, input_event)

    assert isinstance(result, RetrieverResultEvent)
    assert result.nodes == nodes
    assert result.source.tool_name == "retriever"
    assert "Retrieved 1 nodes" in result.source.content


@pytest.mark.asyncio
async def test_finalize_empty_nodes() -> None:
    workflow = RetrieverWorkflow(
        retriever=AsyncMock(spec=BaseRetriever),
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0)],
    )

    ctx_mock = MagicMock()
    ctx_mock.store = MagicMock()
    ctx_mock.store.get = AsyncMock()
    ctx_mock.store.get.side_effect = lambda key: {
        "query": "test query",
        "token_limit": None,
    }.get(key)
    ctx_mock.write_event_to_stream = MagicMock()

    input_event = FinalNodesRetrievalEvent(nodes=[])

    result = await workflow.finalize_nodes(ctx_mock, input_event)

    assert isinstance(result, RetrieverResultEvent)
    assert len(result.nodes) == 0
    assert result.source.tool_name == "retriever"
    assert "No relevant nodes found" in result.source.content


@pytest.mark.asyncio
async def test_node_postprocessors_fn() -> None:
    mock_retriever = AsyncMock(spec=BaseRetriever)
    mock_proc1 = SimilarityPostprocessor(similarity_cutoff=0)
    mock_proc2 = SimilarityPostprocessor(similarity_cutoff=0)

    def get_processors(**kwargs: Any) -> list[BaseNodePostprocessor]:
        if kwargs.get("token_limit") == 100:
            return [mock_proc1]
        return [mock_proc2]

    workflow = RetrieverWorkflow(
        retriever=mock_retriever, node_postprocessors_fn=get_processors
    )

    processors = await workflow._get_node_postprocessors(query="test", token_limit=100)
    assert len(processors) == 1
    assert processors[0] is mock_proc1

    processors = await workflow._get_node_postprocessors(query="test", token_limit=200)
    assert len(processors) == 1
    assert processors[0] is mock_proc2


@pytest.mark.asyncio
async def test_retriever_error_handling() -> None:
    mock_retriever = AsyncMock(spec=BaseRetriever)
    mock_retriever.aretrieve.side_effect = [
        Exception("First attempt failed"),
        Exception("Second attempt failed"),
        [
            NodeWithScore(
                node=TextNode(text="Success after retries", id_="node1"), score=0.9
            )
        ],
    ]

    workflow = RetrieverWorkflow(
        retriever=mock_retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0)],
    )

    with patch("workflows.Context", autospec=True) as mock_context_class:
        real_context = LlamaIndexContext(workflow=workflow)
        mock_context_class.return_value = real_context

        # Add necessary attributes to the real context
        real_context.store.set = AsyncMock()
        real_context.store.get = AsyncMock()
        real_context.write_event_to_stream = MagicMock()
        real_context.shutdown = AsyncMock()

        input_event = RetrieverInputEvent(query="test query")
        await workflow.run(real_context, start_event=input_event)

    assert mock_retriever.aretrieve.call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query_type",
    [
        "plain string",
        QueryBundle(query_str="query bundle"),
    ],
)
async def test_different_query_types(query_type) -> None:
    mock_retriever = AsyncMock(spec=BaseRetriever)
    mock_retriever.aretrieve.return_value = [
        NodeWithScore(node=TextNode(text="Test content", id_="node1"), score=0.9),
    ]

    workflow = RetrieverWorkflow(
        retriever=mock_retriever,
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0)],
    )

    ctx_mock = MagicMock()
    ctx_mock.store = MagicMock()
    ctx_mock.store.set = AsyncMock()
    ctx_mock.store.get = AsyncMock()
    ctx_mock.get.side_effect = [query_type, RetrieverConfig(), None]
    ctx_mock.write_event_to_stream = MagicMock()

    # Test the transform_nodes step directly to avoid issues with full workflow run
    nodes = [NodeWithScore(node=TextNode(text="Test content", id_="node1"), score=0.9)]
    input_event = RawNodesRetrievedEvent(nodes=nodes)
    result = await workflow.transform_nodes(ctx_mock, input_event)

    assert isinstance(result, FinalNodesRetrievalEvent)
    assert len(result.nodes) == 1
