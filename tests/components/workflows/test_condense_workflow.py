from unittest.mock import MagicMock

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import LLM, MockLLM
from llama_index.core.schema import QueryBundle

from private_gpt.components.workflows.others.condenser import (
    CondenseInputEvent,
    CondenseResultEvent,
    CondenserWorkflow,
)


@pytest.fixture
def mock_llm() -> MockLLM:
    llm_mock = MagicMock(spec=LLM, autospec=True)
    llm_mock.metadata.context_window = 4096
    llm_mock.metadata.num_output = 0
    return llm_mock


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
async def test_condense_with_empty_history(mock_llm: MockLLM):
    workflow = CondenserWorkflow(llm=mock_llm)

    query = "What is the meaning of life?"
    input_event = CondenseInputEvent(query=query, chat_history=[])

    result = await workflow.run(start_event=input_event)
    assert isinstance(result, CondenseResultEvent)
    assert result.condensed_query == query
    assert result.original_query == query
    mock_llm.acomplete.assert_not_called()


@pytest.mark.asyncio
async def test_condense_with_history(
    mock_llm: MockLLM, chat_history: list[ChatMessage]
):
    workflow = CondenserWorkflow(llm=mock_llm)

    query = "Can you explain it to me?"
    mock_llm.acomplete.return_value = (
        "What is the meaning of life and how can it be explained?"
    )

    input_event = CondenseInputEvent(query=query, chat_history=chat_history)

    result = await workflow.run(start_event=input_event)
    assert isinstance(result, CondenseResultEvent)
    assert (
        result.condensed_query
        == "What is the meaning of life and how can it be explained?"
    )
    assert result.original_query == query
    mock_llm.acomplete.assert_called_once()


@pytest.mark.asyncio
async def test_condense_with_error(mock_llm: MockLLM, chat_history: list[ChatMessage]):
    workflow = CondenserWorkflow(llm=mock_llm)

    query = "Can you explain it to me?"
    mock_llm.acomplete.side_effect = Exception("LLM Error")

    input_event = CondenseInputEvent(query=query, chat_history=chat_history)

    result = await workflow.run(start_event=input_event)
    assert isinstance(result, CondenseResultEvent)
    assert result.condensed_query == query
    assert result.original_query == query
    mock_llm.acomplete.assert_called_once()


@pytest.mark.asyncio
async def test_condense_with_max_tokens(
    mock_llm: MockLLM, chat_history: list[ChatMessage]
):
    workflow = CondenserWorkflow(llm=mock_llm)

    query = "What's this about?"
    custom_max_tokens = 20

    input_event = CondenseInputEvent(
        query=query, chat_history=chat_history, max_condense_tokens=custom_max_tokens
    )

    await workflow.run(start_event=input_event)
    _, kwargs = mock_llm.acomplete.call_args
    assert kwargs["max_tokens"] == custom_max_tokens


@pytest.mark.asyncio
async def test_condense_with_query_bundle():
    llm = MockLLM()
    workflow = CondenserWorkflow(llm=llm)

    query = QueryBundle(query_str="What is the meaning of life?")
    input_event = CondenseInputEvent(query=query, chat_history=[])

    result = await workflow.run(start_event=input_event)
    assert isinstance(result, CondenseResultEvent)
    assert result.condensed_query == query
    assert result.original_query == query


@pytest.mark.asyncio
@pytest.mark.parametrize("max_condense_tokens", [20, 30, 50])
async def test_condense_with_different_max_tokens(
    mock_llm: MockLLM, chat_history: list[ChatMessage], max_condense_tokens: int
):
    workflow = CondenserWorkflow(llm=mock_llm)

    query = "What's this about?"

    input_event = CondenseInputEvent(
        query=query, chat_history=chat_history, max_condense_tokens=max_condense_tokens
    )

    await workflow.run(start_event=input_event)
    _, kwargs = mock_llm.acomplete.call_args
    assert kwargs["max_tokens"] == max_condense_tokens
