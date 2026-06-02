from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core import QueryBundle
from llama_index.core.base.response.schema import Response
from llama_index.core.schema import NodeWithScore, TextNode

from private_gpt.components.workflows.others.summary_query_engine import (
    SummaryQueryEngine,
)
from private_gpt.components.workflows.others.summary_retriever import (
    InMemoryRetriever,
)

TOOL_CALL_CONTENT = (
    '[0] assistant: [{"tool_id": "0", "tool_name": "knowledge_search", '
    '"tool_kwargs": {"query": "ACC015", "artifacts": ["019d1a93-68fd-772d-800b-ccd6a74807c9"]}}]\n'
    '[1] tool: [{"filename": "transactions_2024.csv", "artifact_id": "019d1a93", '
    '"nodes": [{"content": "| transactionid | accountid | amounteur |\\n| - | - | - |"}]}]'
)


@pytest.fixture
def retriever() -> InMemoryRetriever:
    nodes = [
        NodeWithScore(node=TextNode(id_="node-1", text=TOOL_CALL_CONTENT)),
        NodeWithScore(node=TextNode(id_="node-2", text=TOOL_CALL_CONTENT)),
    ]
    return InMemoryRetriever(nodes)


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.metadata.context_window = 4096
    llm.metadata.num_output = 512
    llm.apredict = AsyncMock(return_value="")  # LLM returns empty — the bug
    llm.astream = AsyncMock(return_value="")
    llm.callback_manager = MagicMock()
    return llm


@pytest.mark.asyncio
async def test_empty_llm_response_drops_all_nodes_and_raises(
    retriever: InMemoryRetriever,
    mock_llm: MagicMock,
) -> None:
    engine = SummaryQueryEngine.from_args(
        retriever=retriever,
        llm=mock_llm,
        use_async=True,
        max_workers=2,
    )

    engine._response_synthesizer.asynthesize = AsyncMock(
        return_value=Response(response="")
    )

    response = await engine.aquery(QueryBundle(query_str="Summarize this"))
    assert isinstance(response, Response)
    assert not response.response


@pytest.mark.asyncio
async def test_empty_llm_response_per_node_all_dropped(
    retriever: InMemoryRetriever,
    mock_llm: MagicMock,
) -> None:
    """Verifies that agenerate_summary_nodes returns None for each node.

    Tests when the per-node synthesizer produces an empty response.
    """
    engine = SummaryQueryEngine.from_args(
        retriever=retriever,
        llm=mock_llm,
        use_async=True,
        max_workers=2,
    )

    engine._response_synthesizer.asynthesize = AsyncMock(
        return_value=Response(response="")
    )

    query_bundle = QueryBundle(query_str="Summarize this")
    node = NodeWithScore(node=TextNode(id_="node-1", text=TOOL_CALL_CONTENT))

    result = await engine.agenerate_summary_nodes(query_bundle, node)
    assert result is None


@pytest.mark.asyncio
async def test_valid_content_produces_summary(
    retriever: InMemoryRetriever,
    mock_llm: MagicMock,
) -> None:
    """Sanity check: when LLM returns a non-empty response, summary is produced."""
    engine = SummaryQueryEngine.from_args(
        retriever=retriever,
        llm=mock_llm,
        use_async=True,
        max_workers=2,
    )

    engine._response_synthesizer.asynthesize = AsyncMock(
        return_value=Response(response="This is a valid summary.")
    )

    query_bundle = QueryBundle(query_str="Summarize this")
    node = NodeWithScore(node=TextNode(id_="node-1", text=TOOL_CALL_CONTENT))

    result = await engine.agenerate_summary_nodes(query_bundle, node)
    assert result is not None
    assert result.get_content() == "This is a valid summary."
