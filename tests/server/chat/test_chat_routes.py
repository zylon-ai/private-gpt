import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from httpx import AsyncClient
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.llms.llm import ToolSelection

from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.chat.input_models import (
    Citations,
    MessageInput,
    ResponseFormat,
    ResponseFormatType,
    System,
    SystemExtensions,
    ToolChoice,
)
from private_gpt.components.llm.custom.mock import FunctionCallingLLMMock
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.tools.tool_names import (
    SEMANTIC_SEARCH_TOOL_NAME,
)
from private_gpt.events.event_errors import Errors
from private_gpt.events.models import (
    FatalError,
    Message,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    SourceBlock,
    TextBlock,
    TextDelta,
    ToolResultBlock,
    ToolUseBlock,
)
from private_gpt.server.chat.chat_router import (
    ChatBody,
)
from private_gpt.server.chat_async.chat_async_service import ChatAsyncService
from private_gpt.server.utils.artifact_input import (
    ArtifactType,
    IngestedArtifact,
)
from private_gpt.settings.settings import settings
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm
from tests.fixtures.mock_injector import MockInjector


def tool_use() -> list[list[str | ToolSelection]]:
    return [
        [
            ToolSelection(
                tool_id=SEMANTIC_SEARCH_TOOL_NAME,
                tool_name=SEMANTIC_SEARCH_TOOL_NAME,
                tool_kwargs={"query": "Lorem ipsum dolor sit amet"},
            ),
        ],
        [
            "Lorem ipsum dolor sit amet",
        ],
    ]


def tools(use_context: bool) -> list[str]:
    return (
        [
            {
                "name": SEMANTIC_SEARCH_TOOL_NAME,
                "type": SEMANTIC_SEARCH_TOOL_NAME + "_v1",
            }
        ]
        if use_context
        else []
    )


def tool_choice(use_context: bool, validation_mode: str = "eager") -> ToolChoice:
    tc = ToolChoice()
    if use_context:
        tc.type = "tool"
        tc.name = SEMANTIC_SEARCH_TOOL_NAME
    tc.validation_mode = validation_mode
    return tc


def tool_context(
    use_context: bool, context_filter: ContextFilter | None = None
) -> list[ArtifactType]:
    if not use_context:
        return []
    return [
        IngestedArtifact(
            context_filter=context_filter or ContextFilter(collection="test")
        )
    ]


async def mock_llm(
    injector: MockInjector, deltas: list[list[str | ToolSelection]] | None = None
) -> None:
    deltas = deltas or tool_use()
    mock_llm = get_mock_function_calling_llm(deltas)

    llm_component = injector.get(LLMComponent)
    llm_component.get_llm = Mock(return_value=mock_llm)
    injector.bind_mock(LLMComponent, llm_component)


@pytest.mark.anyio
async def test_chat_route_produces_a_stream(async_test_client: AsyncClient) -> None:
    body = ChatBody(
        messages=[MessageInput(content="test", role="user")],
        stream=True,
        tools=tools(use_context=False),
        tool_choice=tool_choice(use_context=False),
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    raw_events = response.text.split("\n\n")
    raw_events = [item for item in raw_events if item]
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert len(raw_events) > 0
    assert "stop" in raw_events[-1]


@pytest.mark.anyio
async def test_chat_route_produces_a_single_value(
    async_test_client: AsyncClient,
) -> None:
    body = ChatBody(
        messages=[MessageInput(content="test", role="user")],
        stream=False,
        tools=tools(use_context=False),
        tool_choice=tool_choice(use_context=False),
        tool_context=tool_context(
            use_context=False,
        ),
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    Message.model_validate(response.json())
    assert response.status_code == 200


@pytest.mark.anyio
async def test_chat_with_artifact_context(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    # Ingest an artifact
    collection = str(uuid.uuid4())
    artifact = str(uuid.uuid4())

    ingest_response = await async_test_client.post(
        "/v1/artifacts/ingest",
        json={
            "metadata": {},
            "input": {
                "type": "text",
                "value": "Lorem ipsum dolor sit amet",
            },
            "collection": collection,
            "artifact": artifact,
        },
    )
    assert ingest_response.status_code == 200

    # Mock the LLM to return a tool selection
    await mock_llm(injector)

    # Call the chat route with the artifact context
    body = ChatBody(
        system=System(extensions={SystemExtensions.ZYLON}),
        messages=[MessageInput(content="Lorem ipsum", role="user")],
        stream=False,
        tools=tools(use_context=True),
        tool_choice=tool_choice(use_context=True),
        tool_context=tool_context(
            use_context=True,
            context_filter=ContextFilter(
                collection=collection,
                artifacts=[artifact],
            ),
        ),
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    completion: Message = Message.model_validate(response.json())
    assert response.status_code == 200
    assert completion.content is not None
    assert len(completion.content) == 3
    source_block = next(
        (
            source_block
            for block in completion.content
            if isinstance(block, ToolResultBlock)
            for source_block in block.content
            if isinstance(source_block, SourceBlock)
        ),
        None,
    )
    assert source_block is not None
    assert len(source_block.sources) == 1
    assert source_block.sources[0].document.artifact == artifact

    # Delete the created temp file
    await async_test_client.post(
        "/v1/artifacts/delete",
        json={"collection": collection, "artifact": artifact},
    )


@pytest.mark.anyio
async def test_chat_with_non_existent_artifact_context_fails(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    # Mock the LLM to return a tool selection
    await mock_llm(injector)

    body = ChatBody(
        messages=[MessageInput(content="Lorem ipsum", role="user")],
        stream=False,
        tools=tools(use_context=True),
        tool_choice=tool_choice(use_context=True),
        tool_context=tool_context(
            use_context=True,
            context_filter=ContextFilter(
                collection="test_collection",
                artifacts=["non-existent-artifact"],
            ),
        ),
    )
    result = await async_test_client.post("/v1/messages", json=body.model_dump())
    assert result.status_code == 400
    error = result.json()
    assert "error" in error


@pytest.mark.anyio
async def test_chat_with_non_existent_artifact_context_fails_even_in_lazy_mode(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    # Mock the LLM to return a tool selection
    await mock_llm(injector)

    body = ChatBody(
        messages=[MessageInput(content="Lorem ipsum", role="user")],
        stream=False,
        tools=tools(use_context=True),
        tool_choice=tool_choice(use_context=True, validation_mode="lazy"),
        tool_context=tool_context(
            use_context=True,
            context_filter=ContextFilter(
                collection="test_collection",
                artifacts=["non-existent-artifact"],
            ),
        ),
    )
    result = await async_test_client.post("/v1/messages", json=body.model_dump())
    assert result.status_code == 200

    completion: Message = Message.model_validate(result.json())
    tool_results = [
        block for block in completion.content if isinstance(block, ToolResultBlock)
    ]
    assert len(tool_results) == 1

    tool_result = tool_results[0]
    assert tool_result.is_error


@pytest.mark.anyio
async def test_chat_with_metadata_context(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    # Ingest an artifact
    collection = str(uuid.uuid4())
    artifact = str(uuid.uuid4())
    tag = str(uuid.uuid4())

    ingest_response = await async_test_client.post(
        "/v1/artifacts/ingest",
        json={
            "metadata": {"tag": tag},
            "input": {
                "type": "text",
                "value": "Lorem ipsum dolor sit amet",
            },
            "collection": collection,
            "artifact": artifact,
        },
    )
    assert ingest_response.status_code == 200

    # Mock the LLM to return a tool selection
    await mock_llm(injector)

    # Call the chat route with the tag context
    body = ChatBody(
        system=System(extensions={SystemExtensions.ZYLON}),
        messages=[MessageInput(content="Lorem ipsum", role="user")],
        stream=False,
        tools=tools(use_context=True),
        tool_choice=tool_choice(use_context=True),
        tool_context=tool_context(
            use_context=True,
            context_filter=ContextFilter(
                collection=collection,
                metadata_filter=[{"key": "tag", "operator": "==", "value": tag}],
            ),
        ),
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    completion: Message = Message.model_validate(response.json())
    assert response.status_code == 200
    assert completion.content is not None
    assert len(completion.content) == 3

    source_block = next(
        (
            source_block
            for block in completion.content
            if isinstance(block, ToolResultBlock)
            for source_block in block.content
            if isinstance(source_block, SourceBlock)
        ),
        None,
    )
    assert source_block is not None
    assert len(source_block.sources) == 1
    assert source_block.sources[0].document.artifact == artifact

    # Delete the created temp file
    await async_test_client.post(
        "/v1/artifacts/delete",
        json={"collection": collection, "artifact": artifact},
    )


@pytest.mark.anyio
async def test_chat_with_non_existent_metadata_context_not_fails(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    # Mock the LLM to return a tool selection
    await mock_llm(injector)

    body = ChatBody(
        messages=[MessageInput(content="Lorem ipsum", role="user")],
        stream=False,
        system=System(
            citations=Citations(
                enabled=True,
            )
        ),
        tools=tools(use_context=True),
        tool_choice=tool_choice(use_context=True),
        tool_context=tool_context(
            use_context=True,
            context_filter=ContextFilter(
                collection="test_collection",
                metadata_filter=[
                    {"key": "tag", "operator": "==", "value": "non-existent-tag"}
                ],
            ),
        ),
    )

    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    completion: Message = Message.model_validate(response.json())
    assert response.status_code == 200
    assert completion.content is not None
    assert len(completion.content) == 3

    source_block = next(
        (
            source_block
            for block in completion.content
            if isinstance(block, ToolResultBlock)
            for source_block in block.content
            if isinstance(source_block, SourceBlock)
        ),
        None,
    )
    assert source_block is None or len(source_block.sources) == 0


@pytest.mark.anyio
async def test_chat_with_metadata_and_artifact_context(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    collection = str(uuid.uuid4())

    # Ingest an artifact
    artifact_1 = "test_chat_with_metadata_and_artifact_1"
    ingest_response = await async_test_client.post(
        "/v1/artifacts/ingest",
        json={
            "metadata": {"tag": "test_chat_with_metadata_and_artifact_tag_1"},
            "input": {
                "type": "text",
                "value": "Lorem ipsum dolor sit amet",
            },
            "collection": collection,
            "artifact": artifact_1,
        },
    )
    assert ingest_response.status_code == 200

    # Ingest a second artifact
    artifact_2 = "test_chat_with_metadata_and_artifact_2"
    ingest_response = await async_test_client.post(
        "/v1/artifacts/ingest",
        json={
            "metadata": {"tag": "test_chat_with_metadata_and_artifact_tag_2"},
            "input": {
                "type": "text",
                "value": "Lorem ipsum dolor sit amet",
            },
            "collection": collection,
            "artifact": artifact_2,
        },
    )
    assert ingest_response.status_code == 200

    # Mock the LLM to return a tool selection
    await mock_llm(injector)

    # Completions with the first artifact's id and the second artifact's metadata
    body = ChatBody(
        system=System(
            citations=Citations(
                enabled=True,
            ),
            extensions={SystemExtensions.ZYLON},
        ),
        messages=[MessageInput(content="Lorem ipsum", role="user")],
        stream=False,
        tools=tools(use_context=True),
        tool_choice=tool_choice(use_context=True),
        tool_context=tool_context(
            use_context=True,
            context_filter=ContextFilter(
                collection=collection,
                artifacts=[artifact_1],
                metadata_filter=[
                    {
                        "key": "tag",
                        "operator": "==",
                        "value": "test_chat_with_metadata_and_artifact_tag_2",
                    }
                ],
            ),
        ),
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    completion: Message = Message.model_validate(response.json())
    assert response.status_code == 200
    assert len(completion.content) > 1

    source_block = next(
        (
            source_block
            for block in completion.content
            if isinstance(block, ToolResultBlock)
            for source_block in block.content
            if isinstance(source_block, SourceBlock)
        ),
        None,
    )
    assert source_block is not None
    assert source_block.sources is not None

    # Assert both artifacts are part of the used sources
    source_artifacts: set = {
        source.document.artifact for source in source_block.sources
    }
    assert source_artifacts.issubset({artifact_1, artifact_2})

    # Delete the created temp files
    await async_test_client.post(
        "/v1/artifacts/delete",
        json={"collection": collection, "artifact": artifact_1},
    )
    await async_test_client.post(
        "/v1/artifacts/delete",
        json={"collection": collection, "artifact": artifact_2},
    )


@pytest.mark.anyio
async def test_chat_multitenancy_isolation(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    # Ingest an artifact in collection 1 and another in collection 2

    collection_1 = str(uuid.uuid4())
    artifact_1 = str(uuid.uuid4())

    ingest_response = await async_test_client.post(
        "/v1/artifacts/ingest",
        json={
            "metadata": {},
            "input": {
                "type": "text",
                "value": "Lorem ipsum dolor sit amet",
            },
            "collection": collection_1,
            "artifact": artifact_1,
        },
    )

    assert ingest_response.status_code == 200

    collection_2 = str(uuid.uuid4())
    artifact_2 = str(uuid.uuid4())
    ingest_response = await async_test_client.post(
        "/v1/artifacts/ingest",
        json={
            "metadata": {},
            "input": {
                "type": "text",
                "value": "Lorem ipsum dolor sit amet",
            },
            "collection": collection_2,
            "artifact": artifact_2,
        },
    )

    assert ingest_response.status_code == 200

    # Mock the LLM to return a tool selection (with 2 blocks)
    await mock_llm(injector)

    # Completions with the first artifact's id (from collection 1)
    body = ChatBody(
        system=System(
            citations=Citations(
                enabled=True,
            ),
            extensions={SystemExtensions.ZYLON},
        ),
        messages=[MessageInput(content="Lorem ipsum", role="user")],
        stream=False,
        tools=tools(use_context=True),
        tool_choice=tool_choice(use_context=True),
        tool_context=tool_context(
            use_context=True,
            context_filter=ContextFilter(
                collection=collection_1,
                artifacts=[artifact_1],
            ),
        ),
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    completion: Message = Message.model_validate(response.json())
    assert response.status_code == 200
    assert completion.content is not None
    assert len(completion.content) == 3

    source_block = next(
        (
            source_block
            for block in completion.content
            if isinstance(block, ToolResultBlock)
            for source_block in block.content
            if isinstance(source_block, SourceBlock)
        ),
        None,
    )
    assert source_block is not None
    assert isinstance(source_block, SourceBlock)

    sources_with_text = [source for source in source_block.sources if source.text]
    assert len(sources_with_text) == 1
    assert sources_with_text[0].document.artifact == artifact_1
    assert sources_with_text[0].document.doc_metadata is not None
    assert sources_with_text[0].document.doc_metadata["collection"] == collection_1

    # Mock the LLM to return a tool selection (with 2 blocks)
    await mock_llm(injector)

    # Completions with the first artifact's id (from collection 2)
    body = ChatBody(
        system=System(
            citations=Citations(
                enabled=True,
            ),
            extensions={SystemExtensions.ZYLON},
        ),
        messages=[MessageInput(content="Lorem ipsum", role="user")],
        stream=False,
        tools=tools(use_context=True),
        tool_choice=tool_choice(use_context=True),
        tool_context=tool_context(
            use_context=True,
            context_filter=ContextFilter(
                collection=collection_2,
                artifacts=[artifact_2],
            ),
        ),
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    completion: Message = Message.model_validate(response.json())
    assert response.status_code == 200
    assert completion.content is not None
    assert len(completion.content) == 3

    source_block = next(
        (
            source_block
            for block in completion.content
            if isinstance(block, ToolResultBlock)
            for source_block in block.content
            if isinstance(source_block, SourceBlock)
        ),
        None,
    )
    assert source_block is not None
    assert isinstance(source_block, SourceBlock)

    sources_with_text = [source for source in source_block.sources if source.text]
    assert len(sources_with_text) == 1
    assert sources_with_text[0].document.artifact == artifact_2
    assert sources_with_text[0].document.doc_metadata is not None
    assert sources_with_text[0].document.doc_metadata["collection"] == collection_2

    # Completions with both artifacts' ids (crash)
    body = ChatBody(
        messages=[MessageInput(content="Lorem ipsum", role="user")],
        system=System(
            citations=Citations(
                enabled=True,
            )
        ),
        stream=False,
        tools=tools(use_context=True),
        tool_choice=tool_choice(use_context=True),
        tool_context=tool_context(
            use_context=True,
            context_filter=ContextFilter(
                collection=collection_1,
                artifacts=[artifact_1, artifact_2],
            ),
        ),
    )

    response = await async_test_client.post("/v1/messages", json=body.model_dump())
    assert response.status_code == 400
    error = response.json()
    assert "error" in error


async def test_chat_with_default_prompt(async_test_client: AsyncClient) -> None:
    body = ChatBody(
        messages=[MessageInput(content="test", role="user")],
        system=System(
            use_default_prompt=True,
        ),
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    completion: Message = Message.model_validate(response.json())
    assert response.status_code == 200
    assert "You are" in completion.content[0].text


@pytest.mark.anyio
async def test_chat_without_default_prompt(async_test_client: AsyncClient) -> None:
    body = ChatBody(
        messages=[MessageInput(content="test", role="user")],
        system=System(
            use_default_prompt=False,
        ),
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    completion: Message = Message.model_validate(response.json())
    assert response.status_code == 200
    assert "Utilize Training Data" not in completion.content[0].text


@pytest.mark.anyio
async def test_chat_with_a_very_long_message(async_test_client: AsyncClient) -> None:
    body = ChatBody(
        messages=[MessageInput(content="a" * 30000, role="user")],
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())
    completion: FatalError = FatalError.model_validate(response.json())

    assert response.status_code == 413
    assert completion.type == "error"
    assert completion.error.type == Errors.Types.REQUEST_TOO_LARGE_ERROR.value
    assert completion.error.detail is not None
    assert completion.error.detail.code


@pytest.mark.anyio
async def test_chat_with_a_very_long_message_in_streaming(
    async_test_client: AsyncClient,
) -> None:
    body = ChatBody(
        messages=[MessageInput(content="a" * 30000, role="user")],
        stream=True,
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())
    raw_events = response.text.split("\n\n")
    raw_events = [item for item in raw_events if item]
    assert raw_events
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert len(raw_events) > 0
    error_event = next(item for item in raw_events if "event: error" in item)
    payload = json.loads(error_event.split("data: ", 1)[1])
    assert payload["type"] == "error"
    assert (
        payload["error"]["detail"]["code"]
        == Errors.Codes.REQUEST_TOO_LARGE_USER_MSG.value
    )


@pytest.mark.anyio
async def test_chat_body_validation_empty_messages(
    async_test_client: AsyncClient,
) -> None:
    body = {
        "messages": [],
        "stream": False,
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 400
    payload = response.json()
    completion: FatalError = FatalError.model_validate(payload)
    assert completion.type == "error"
    assert completion.error.type == Errors.Types.INVALID_REQUEST_ERROR.value
    assert completion.error.detail is not None
    assert completion.error.detail.code == Errors.Codes.INVALID_REQUEST_ERROR.value
    assert completion.error.detail.explanation == payload["detail"]
    error_detail = payload["detail"]
    assert any("Messages cannot be empty" in str(err) for err in error_detail)


@pytest.mark.anyio
async def test_chat_body_validation_agent_mode_with_tools(
    async_test_client: AsyncClient,
) -> None:
    body = {
        "messages": [{"content": "test", "role": "user"}],
        "tools": [
            {
                "name": "test_tool",
                "description": "A test tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query parameter",
                        }
                    },
                    "required": ["query"],
                },
            }
        ],
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 200

    body = {
        "messages": [{"content": "test", "role": "user"}],
        "tools": [
            {
                "name": SEMANTIC_SEARCH_TOOL_NAME,
                "type": SEMANTIC_SEARCH_TOOL_NAME + "_v1",
            }
        ],
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 400


@pytest.mark.anyio
async def test_chat_body_validation_duplicate_tools(
    async_test_client: AsyncClient,
) -> None:
    body = {
        "messages": [{"content": "test", "role": "user"}],
        "tools": [
            {
                "name": "test_tool",
                "description": "A test tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query parameter",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "test_tool",
                "description": "A test tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query parameter",
                        }
                    },
                    "required": ["query"],
                },
            },
        ],
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 400

    body = {
        "messages": [{"content": "test", "role": "user"}],
        "tools": [
            {
                "name": SEMANTIC_SEARCH_TOOL_NAME,
                "type": SEMANTIC_SEARCH_TOOL_NAME + "_v1",
            },
            {
                "name": SEMANTIC_SEARCH_TOOL_NAME,
                "type": SEMANTIC_SEARCH_TOOL_NAME + "_v1",
            },
        ],
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 400

    body = {
        "messages": [{"content": "test", "role": "user"}],
        "tools": [
            {
                "name": SEMANTIC_SEARCH_TOOL_NAME,
                "type": SEMANTIC_SEARCH_TOOL_NAME + "_v1",
            },
            {
                "name": SEMANTIC_SEARCH_TOOL_NAME,
                "description": "A test tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query parameter",
                        }
                    },
                    "required": ["query"],
                },
            },
        ],
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 400


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("messages", "expected_error"),
    [
        (
            [
                {"content": "test", "role": "user"},
            ],
            None,
        ),
        (
            [
                {"content": "test", "role": "user"},
                {"content": "response", "role": "assistant"},
            ],
            None,
        ),
        (
            [
                {"content": "test", "role": "system"},
            ],
            "Messages cannot be empty",
        ),
    ],
)
async def test_chat_body_validation_invalid_last_message_role(
    async_test_client: AsyncClient,
    messages: list,
    expected_error: str | None,
) -> None:
    body = {
        "messages": messages,
    }
    route = "/v1/messages"
    response = await async_test_client.post(route, json=body)
    if expected_error is None:
        assert response.status_code == 200
        return
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert any(expected_error in str(err) for err in error_detail)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("role", "content", "expected_error"),
    [
        (
            "user",
            [
                {"type": "text", "text": "test"},
                {"type": "tool_use", "id": "tool_1", "name": "test_tool", "input": {}},
            ],
            "Tool use blocks can only be used in assistant messages",
        ),
    ],
)
async def test_chat_body_validation_invalid_block_role(
    async_test_client: AsyncClient, role: str, content: list, expected_error: str
) -> None:
    body = {"messages": [{"role": role, "content": content}]}
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert any(expected_error in str(err) for err in error_detail)


@pytest.mark.anyio
async def test_chat_body_validation_duplicate_tool_use_id(
    async_test_client: AsyncClient,
) -> None:
    body = {
        "messages": [
            {"content": "test", "role": "user"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "test_tool",
                        "input": {},
                    },
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "test_tool2",
                        "input": {},
                    },
                ],
            },
            {"content": "continue", "role": "user"},
        ]
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert any("Duplicate tool use ID found" in str(err) for err in error_detail)


@pytest.mark.anyio
async def test_chat_body_validation_tool_result_references_unknown_tool(
    async_test_client: AsyncClient,
) -> None:
    body = {
        "messages": [
            {"content": "test", "role": "user"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "test_tool",
                        "input": {},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_2",
                        "content": "result",
                    },
                ],
            },
        ]
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert any(
        "Tool result block references an unknown tool use ID" in str(err)
        for err in error_detail
    )


@pytest.mark.anyio
async def test_chat_body_validation_mismatched_tool_ids(
    async_test_client: AsyncClient,
) -> None:
    body = {
        "messages": [
            {"content": "test", "role": "user"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "test_tool",
                        "input": {},
                    },
                    {
                        "type": "tool_use",
                        "id": "tool_2",
                        "name": "test_tool2",
                        "input": {},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_1",
                        "content": "result",
                    },
                ],
            },
        ]
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert any(
        "Tool result blocks must match the tool use IDs" in str(err)
        for err in error_detail
    )


@pytest.mark.anyio
async def test_chat_body_validation_none_block(async_test_client: AsyncClient) -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "test"},
                    None,
                ],
            }
        ],
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 400


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("messages", "expected_status_error"),
    [
        (
            [
                {"content": "test", "role": "user"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me help you"},
                        {
                            "type": "tool_use",
                            "id": "tool_1",
                            "name": "test_tool",
                            "input": {"query": "test"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "Tool result here",
                        },
                    ],
                },
            ],
            None,
        ),
        (
            [
                {"content": "test", "role": "user"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tool_1",
                            "name": "test_tool",
                            "input": {},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tool_1",
                            "content": "result",
                        },
                    ],
                },
            ],
            None,
        ),
        (
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"},
                    ],
                }
            ],
            None,
        ),
    ],
)
async def test_chat_body_validation_valid_scenarios(
    async_test_client: AsyncClient, messages: list, expected_status_error: int | None
) -> None:
    body = {"messages": messages}
    route = "/v1/messages"
    response = await async_test_client.post(route, json=body)
    expected_status = 200 if expected_status_error is None else expected_status_error
    assert response.status_code == expected_status, (
        f"Expected {expected_status} but got {response.status_code}"
        f"Response: {response.json()}"
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("tool_choice", "expected_output_agent", "expected_output_normal"),
    [
        ("auto", 200, 200),
        ("test_tool", 200, 200),
        ("non_existent_tool", 200, 400),
    ],
)
async def test_chat_body_validation_tool_choice_options(
    async_test_client: AsyncClient,
    tool_choice: str,
    expected_output_agent: int,
    expected_output_normal: int,
) -> None:
    for agent_mode in [True, False]:
        body = {
            "messages": [{"content": "test", "role": "user"}],
            "tool_choice": {
                "type": "auto" if tool_choice == "auto" else "tool",
                "name": tool_choice if tool_choice != "auto" else None,
            },
            "tools": [
                {
                    "name": "test_tool",
                    "description": "A test tool",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query parameter",
                            }
                        },
                        "required": ["query"],
                    },
                }
            ]
            if not agent_mode
            else None,
        }
        route = "/v1/messages"
        response = await async_test_client.post(route, json=body)
        expected = expected_output_agent if agent_mode else expected_output_normal
        assert response.status_code == expected, (
            f"Expected {expected} for agent_mode={agent_mode}, "
            f"tool_choice='{tool_choice}', but got {response.status_code}"
            f"Response: {response.json()}"
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "context_filter",
    [
        {"collection": "test"},
        {"artifacts": ["artifact1", "artifact2"]},
        {"metadata_filter": [{"key": "type", "operator": "==", "value": "test"}]},
        {
            "collection": "test",
            "artifacts": ["artifact1"],
            "metadata_filter": [{"key": "type", "operator": "==", "value": "test"}],
        },
    ],
)
async def test_chat_body_validation_context_filter_combinations(
    async_test_client: AsyncClient, context_filter: dict
) -> None:
    body = {
        "messages": [{"content": "test", "role": "user"}],
        "tools": [
            {
                "name": SEMANTIC_SEARCH_TOOL_NAME,
                "type": SEMANTIC_SEARCH_TOOL_NAME + "_v1",
            }
        ],
        "tool_context": [
            {
                "type": "ingested_artifact",
                "context_filter": context_filter,
            }
        ],
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code != 422


@pytest.mark.anyio
async def test_chat_with_tools_with_schemas_with_underscore(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    def tool_use() -> list[list[str | ToolSelection]]:
        return [
            [
                ToolSelection(
                    tool_id="test_tool",
                    tool_name="test_tool",
                    tool_kwargs={"_query": "Lorem ipsum dolor sit amet"},
                ),
            ],
        ]

    await mock_llm(injector, deltas=tool_use())

    body = {
        "messages": [{"content": "test", "role": "user"}],
        "tools": [
            {
                "name": "test_tool",
                "description": "A test tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "_query": {
                            "type": "string",
                            "description": "The query parameter",
                        }
                    },
                    "required": ["query"],
                },
            }
        ],
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 200

    completion: Message = Message.model_validate(response.json())
    assert completion.content is not None
    assert len(completion.content) == 1

    assert isinstance(completion.content[0], ToolUseBlock)


@pytest.mark.anyio
async def test_chat_completion_with_json_schema_response_format(
    async_test_client: AsyncClient,
) -> None:
    """Test chat completion with structured JSON output."""
    body = ChatBody(
        messages=[MessageInput(content="Generate a user profile", role="user")],
        response_format=ResponseFormat(
            type=ResponseFormatType.json_schema,
            json_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"},
                    "email": {"type": "string"},
                },
                "required": ["name", "age"],
            },
        ),
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    assert response.status_code == 200
    completion = Message.model_validate(response.json())
    assert completion.content is not None
    assert len(completion.content) == 1
    assert isinstance(completion.content[0], TextBlock)


@pytest.mark.anyio
async def test_chat_completion_with_json_schema_rejects_tools(
    async_test_client: AsyncClient,
) -> None:
    """Test that JSON schema response format rejects tools."""
    body = {
        "model": None,
        "messages": [{"content": "test", "role": "user"}],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "type": "object",
                "properties": {"result": {"type": "string"}},
            },
        },
        "tools": [
            {
                "name": "test_tool",
                "description": "A test tool",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ],
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert any(
        "Tools are not supported when response_format is set to json_schema" in str(err)
        for err in error_detail
    )


@pytest.mark.anyio
async def test_chat_completion_with_system_prompt_injection(
    async_test_client: AsyncClient,
) -> None:
    """Test that system prompt is properly injected into messages."""
    body = ChatBody(
        messages=[MessageInput(content="Hello", role="user")],
        system=System(
            text="You are a helpful assistant specialized in mathematics.",
            use_default_prompt=False,
        ),
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    assert response.status_code == 200
    completion = Message.model_validate(response.json())
    assert completion.content is not None


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("temperature", "top_p", "top_k", "max_tokens"),
    [
        (0.7, 0.9, 50, 1000),
        (0.1, 0.95, 40, 500),
        (1.0, 1.0, 100, 2000),
    ],
)
async def test_chat_completion_with_sampling_parameters(
    async_test_client: AsyncClient,
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
) -> None:
    """Test chat completion with various sampling parameters."""
    body = ChatBody(
        messages=[MessageInput(content="Tell me a story", role="user")],
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=42,
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    assert response.status_code == 200
    completion = Message.model_validate(response.json())
    assert completion.content is not None


@pytest.mark.anyio
async def test_chat_completion_with_tool_context_direct_call(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    """Test chat completion with tool context for direct function calls."""
    tool_deltas = [
        [
            ToolSelection(
                tool_id="calculator",
                tool_name="calculator",
                tool_kwargs={"expression": "2 + 2"},
            )
        ],
        ["4"],
    ]
    await mock_llm(injector, deltas=tool_deltas)

    body = ChatBody(
        messages=[MessageInput(content="Calculate 2 + 2", role="user")],
        tools=[
            {
                "name": "calculator",
                "description": "Perform calculations",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                    },
                    "required": ["expression"],
                },
            }
        ],
        tool_choice=ToolChoice(type="tool", name="calculator"),
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    assert response.status_code == 200
    completion = Message.model_validate(response.json())
    assert completion.content is not None
    assert any(isinstance(block, ToolUseBlock) for block in completion.content)
    assert not any(isinstance(block, ToolResultBlock) for block in completion.content)


@pytest.mark.anyio
async def test_chat_completion_streaming_with_citations(
    async_test_client: AsyncClient,
) -> None:
    """Test streaming chat completion with citations enabled."""
    body = ChatBody(
        messages=[MessageInput(content="Explain quantum physics", role="user")],
        stream=True,
        system=System(citations=Citations(enabled=True)),
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    raw_events = response.text.split("\n\n")
    raw_events = [item for item in raw_events if item and item.strip()]
    assert len(raw_events) > 0


@pytest.mark.anyio
async def test_chat_completion_with_parallel_tool_calls_disabled(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    """Test chat completion with parallel tool calls disabled."""
    tool_deltas = [
        [
            ToolSelection(
                tool_id="tool1",
                tool_name="tool1",
                tool_kwargs={"param": "value1"},
            )
        ],
        ["Result from tool1"],
    ]
    await mock_llm(injector, deltas=tool_deltas)

    body = ChatBody(
        messages=[MessageInput(content="Use multiple tools", role="user")],
        tools=[
            {
                "name": "tool1",
                "description": "First tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                },
            },
            {
                "name": "tool2",
                "description": "Second tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                },
            },
        ],
        tool_choice=ToolChoice(disable_parallel_tool_use=True),
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    assert response.status_code == 200, response.text
    completion = Message.model_validate(response.json())
    assert completion.content is not None


@pytest.mark.anyio
async def test_chat_completion_with_conversation_history_and_tools(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    """Test chat completion with multi-turn conversation including tool usage."""
    tool_deltas = [
        [
            ToolSelection(
                tool_id="search",
                tool_name="search",
                tool_kwargs={"query": "weather"},
            )
        ],
        ["Current weather data retrieved successfully."],
    ]
    await mock_llm(injector, deltas=tool_deltas)

    body = ChatBody(
        messages=[
            MessageInput(content="Hello", role="user"),
            MessageInput(content="Hi! How can I help you?", role="assistant"),
            MessageInput(content="What's the weather like?", role="user"),
            MessageInput(
                content=[
                    ToolUseBlock(
                        id="tool_1",
                        name="search",
                        input={"query": "weather"},
                    ),
                    ToolResultBlock(
                        tool_use_id="tool_1",
                        content="Sunny, 72°F",
                    ),
                    TextBlock(text="The weather is sunny and 72°F."),
                ],
                role="assistant",
            ),
            MessageInput(
                content="Thanks! Can you check tomorrow's forecast?", role="user"
            ),
        ],
        tools=[
            {
                "name": "search",
                "description": "Search for information",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    assert response.status_code == 200, response.text
    completion = Message.model_validate(response.json())
    assert completion.content is not None


@pytest.mark.anyio
async def test_agent_completion_with_context_and_tool_interaction(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    """Test agent completion with context search and tool interaction."""
    collection = str(uuid.uuid4())
    artifact = str(uuid.uuid4())

    # Ingest test data
    ingest_response = await async_test_client.post(
        "/v1/artifacts/ingest",
        json={
            "metadata": {"type": "technical_doc"},
            "input": {
                "type": "text",
                "value": "FastAPI is a modern web framework for building APIs with Python.",
            },
            "collection": collection,
            "artifact": artifact,
        },
    )
    assert ingest_response.status_code == 200

    tool_deltas = [
        [
            ToolSelection(
                tool_id=SEMANTIC_SEARCH_TOOL_NAME,
                tool_name=SEMANTIC_SEARCH_TOOL_NAME,
                tool_kwargs={"query": "FastAPI framework"},
            )
        ],
        ["FastAPI is a modern web framework for building APIs with Python."],
    ]
    await mock_llm(injector, deltas=tool_deltas)

    body = ChatBody(
        system=System(
            citations=Citations(
                enabled=True,
            ),
            extensions={SystemExtensions.ZYLON},
        ),
        messages=[MessageInput(content="Tell me about FastAPI", role="user")],
        tools=tools(use_context=True),
        tool_choice=ToolChoice(type="tool", name=SEMANTIC_SEARCH_TOOL_NAME),
        tool_context=[
            IngestedArtifact(
                context_filter=ContextFilter(
                    collection=collection,
                    metadata_filter=[
                        {"key": "type", "operator": "==", "value": "technical_doc"}
                    ],
                )
            )
        ],
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    assert response.status_code == 200
    completion = Message.model_validate(response.json())
    assert completion.content is not None
    assert len(completion.content) >= 3  # Tool use, tool result, text response

    # Verify tool result contains source information
    tool_result_block = next(
        (block for block in completion.content if isinstance(block, ToolResultBlock)),
        None,
    )
    assert tool_result_block is not None
    source_block = next(
        (
            block
            for block in tool_result_block.content
            if isinstance(block, SourceBlock)
        ),
        None,
    )
    assert source_block is not None
    assert len(source_block.sources) > 0

    # Cleanup
    await async_test_client.post(
        "/v1/artifacts/delete",
        json={"collection": collection, "artifact": artifact},
    )


@pytest.mark.anyio
async def test_agent_completion_with_complex_metadata_filtering(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    """Test agent completion with complex metadata filtering scenarios."""
    collection = str(uuid.uuid4())

    # Ingest multiple documents with different metadata
    documents = [
        {
            "artifact": f"doc_{i}",
            "text": f"Document {i} content about topic {i % 3}",
            "metadata": {
                "category": "research" if i % 2 == 0 else "general",
                "priority": i % 5,
                "tags": [f"tag_{i}", f"tag_{i % 3}"],
            },
        }
        for i in range(10)
    ]

    for doc in documents:
        ingest_response = await async_test_client.post(
            "/v1/artifacts/ingest",
            json={
                "metadata": doc["metadata"],
                "input": {
                    "type": "text",
                    "value": doc["text"],
                },
                "collection": collection,
                "artifact": str(doc["artifact"]),
            },
        )
        assert ingest_response.status_code == 200

    await mock_llm(injector)

    # Test with complex metadata filter
    body = ChatBody(
        messages=[
            MessageInput(
                content="Find research documents with high priority", role="user"
            )
        ],
        tools=[SEMANTIC_SEARCH_TOOL_NAME],
        tool_choice=ToolChoice(type="tool", name=SEMANTIC_SEARCH_TOOL_NAME),
        tool_context=[
            IngestedArtifact(
                context_filter=ContextFilter(
                    collection=collection,
                    metadata_filter=[
                        {"key": "category", "operator": "==", "value": "research"},
                        {"key": "priority", "operator": ">=", "value": 3},
                        {"key": "tags", "operator": "==", "value": "tag_1"},
                    ],
                )
            )
        ],
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    assert response.status_code == 200
    completion = Message.model_validate(response.json())
    assert completion.content is not None

    # Cleanup
    for doc in documents:
        await async_test_client.post(
            "/v1/artifacts/delete",
            json={"collection": collection, "artifact": doc["artifact"]},
        )


@pytest.mark.anyio
async def test_agent_completion_with_mcp_servers_fails(
    async_test_client: AsyncClient,
) -> None:
    """Test that agent mode rejects MCP servers."""
    body = {
        "messages": [{"content": "test", "role": "user"}],
        "mcp_servers": [
            {
                "url": "http://localhost:8080/mcp",
                "tool_configuration": {"enabled": True},
            }
        ],
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 400


@pytest.mark.anyio
async def test_chat_completion_with_empty_content_fails(
    async_test_client: AsyncClient,
) -> None:
    """Test that empty message content fails validation."""
    body = {
        "messages": [{"content": "", "role": "user"}],
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 400
    error_detail = response.json()["detail"]
    assert any("Message content cannot be empty" in str(err) for err in error_detail)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("frequency_penalty", "presence_penalty", "repetition_penalty"),
    [
        (0.1, 0.2, 1.1),
        (0.5, 0.0, 1.0),
        (0.0, 0.8, 1.2),
    ],
)
async def test_chat_completion_with_penalty_parameters(
    async_test_client: AsyncClient,
    frequency_penalty: float,
    presence_penalty: float,
    repetition_penalty: float,
) -> None:
    """Test chat completion with various penalty parameters."""
    body = ChatBody(
        messages=[MessageInput(content="Write a creative story", role="user")],
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        repetition_penalty=repetition_penalty,
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    assert response.status_code == 200
    completion: Message = Message.model_validate(response.json())
    assert completion.content is not None


@pytest.mark.anyio
async def test_agent_completion_requires_tool_context_for_internal_tools(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    """Test that agent mode requires tool context for internal tools."""
    # Case 1. A internal tool is provided without tool context
    body = {
        "messages": [{"content": "test", "role": "user"}],
        "tools": [
            {
                "name": SEMANTIC_SEARCH_TOOL_NAME,
                "type": SEMANTIC_SEARCH_TOOL_NAME + "_v1",
            }
        ],
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 400

    # Case 2. An external tool is provided without tool context
    def tool_use() -> list[list[str | ToolSelection]]:
        return [
            [
                ToolSelection(
                    tool_id=SEMANTIC_SEARCH_TOOL_NAME,
                    tool_name=SEMANTIC_SEARCH_TOOL_NAME,
                    tool_kwargs={"_query": "Lorem ipsum dolor sit amet"},
                )
            ],
        ]

    await mock_llm(injector, deltas=tool_use())

    body = {
        "messages": [{"content": "test", "role": "user"}],
        "tools": [
            {
                "name": SEMANTIC_SEARCH_TOOL_NAME,
                "description": "A test tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "_query": {
                            "type": "string",
                            "description": "The query parameter",
                        }
                    },
                    "required": ["_query"],
                },
            }
        ],
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 200

    completion: Message = Message.model_validate(response.json())
    assert completion.content is not None
    assert len(completion.content) == 1
    assert isinstance(completion.content[0], ToolUseBlock)

    # Case 3. An internal tool is provided with tool context
    def tool_use() -> list[list[str | ToolSelection]]:
        return [
            [
                ToolSelection(
                    tool_id=SEMANTIC_SEARCH_TOOL_NAME,
                    tool_name=SEMANTIC_SEARCH_TOOL_NAME,
                    tool_kwargs={"query": "Lorem ipsum dolor sit amet"},
                )
            ],
            ["4"],
        ]

    await mock_llm(injector, deltas=tool_use())

    body = {
        "messages": [{"content": "test", "role": "user"}],
        "tools": [
            {
                "name": SEMANTIC_SEARCH_TOOL_NAME,
                "type": SEMANTIC_SEARCH_TOOL_NAME + "_v1",
            }
        ],
        "tool_context": [
            {
                "type": "ingested_artifact",
                "context_filter": {"collection": "test_collection"},
            }
        ],
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 200

    completion: Message = Message.model_validate(response.json())
    assert completion.content is not None
    assert len(completion.content) >= 1
    assert any(isinstance(block, ToolUseBlock) for block in completion.content)
    assert any(isinstance(block, ToolResultBlock) for block in completion.content)


@pytest.mark.anyio
async def test_chat_completion_with_min_p_parameter(
    async_test_client: AsyncClient,
) -> None:
    """Test chat completion with min_p sampling parameter."""
    body = ChatBody(
        messages=[MessageInput(content="Explain machine learning", role="user")],
        min_p=0.05,
        temperature=0.8,
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    assert response.status_code == 200
    completion: Message = Message.model_validate(response.json())
    assert completion.content is not None


async def mock_slow_stream_generator():
    """Mock generator that yields events slowly to simulate streaming."""
    yield RawContentBlockStartEvent(block_id="1", content_block=TextBlock(text=""))
    await asyncio.sleep(1)
    yield RawContentBlockDeltaEvent(block_id="1", delta=TextDelta(text="Hello"))
    await asyncio.sleep(1)
    yield RawContentBlockDeltaEvent(block_id="1", delta=TextDelta(text=" world!"))
    await asyncio.sleep(1)
    yield RawContentBlockStopEvent(block_id="1")


@pytest.mark.anyio
async def test_chat_handles_client_disconnection_non_streaming(
    async_test_client: AsyncClient, injector
) -> None:
    chat_service = injector.get(ChatAsyncService)
    original_cancel_stream = chat_service.cancel_stream
    chat_service.cancel_stream = AsyncMock(side_effect=original_cancel_stream)
    chat_service.get_stream_events = AsyncMock(
        return_value=mock_slow_stream_generator()
    )

    body = ChatBody(
        messages=[MessageInput(content="What is Python?", role="user")],
        stream=False,
    )

    request_task = asyncio.create_task(
        async_test_client.post("/v1/messages", json=body.model_dump())
    )

    start_time = asyncio.get_event_loop().time()
    start_time = asyncio.get_running_loop().time()
    while asyncio.get_running_loop().time() - start_time < 5.0:
        if chat_service.get_stream_events.called:
            request_task.cancel()
            break
        await asyncio.sleep(0.1)
    else:
        request_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await request_task
    assert chat_service.cancel_stream.called


@pytest.mark.anyio
async def test_chat_handles_client_disconnection_streaming(
    async_test_client: AsyncClient, injector
) -> None:
    chat_service = injector.get(ChatAsyncService)
    original_cancel_stream = chat_service.cancel_stream
    chat_service.cancel_stream = AsyncMock(side_effect=original_cancel_stream)
    chat_service.get_stream_events = AsyncMock(
        return_value=mock_slow_stream_generator()
    )

    body = ChatBody(
        messages=[MessageInput(content="Tell me a story", role="user")],
        stream=True,
    )

    request_task = asyncio.create_task(
        async_test_client.post("/v1/messages", json=body.model_dump())
    )

    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < 5.0:
        if chat_service.get_stream_events.called:
            request_task.cancel()
            break
        await asyncio.sleep(0.1)
    else:
        request_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await request_task
    assert chat_service.cancel_stream.called


@pytest.mark.anyio
async def test_validate_chat_with_invalid_tool_context_fails(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:

    body = ChatBody(
        messages=[MessageInput(content="Lorem ipsum", role="user")],
        stream=False,
        system=System(
            citations=Citations(
                enabled=True,
            )
        ),
        tools=tools(use_context=True),
        tool_choice=tool_choice(use_context=True),
        tool_context=tool_context(
            use_context=True,
            context_filter=ContextFilter(
                collection="test_collection",
                artifacts=["non_existent_artifact"],
            ),
        ),
    )

    response = await async_test_client.post(
        "/v1/messages/validate", json=body.model_dump()
    )
    assert response.status_code == 400, response.json()


@pytest.mark.anyio
async def test_validate_chat_with_valid_tool_context_not_fails(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    # Ingest an artifact
    collection = str(uuid.uuid4())
    artifact = str(uuid.uuid4())
    tag = str(uuid.uuid4())

    ingest_response = await async_test_client.post(
        "/v1/artifacts/ingest",
        json={
            "metadata": {"tag": tag},
            "input": {
                "type": "text",
                "value": "Lorem ipsum dolor sit amet",
            },
            "collection": collection,
            "artifact": artifact,
        },
    )
    assert ingest_response.status_code == 200

    # Call the validate chat route with the tag context
    body = ChatBody(
        messages=[MessageInput(content="Lorem ipsum", role="user")],
        stream=False,
        system=System(
            citations=Citations(
                enabled=True,
            )
        ),
        tools=tools(use_context=True),
        tool_choice=tool_choice(use_context=True),
        tool_context=tool_context(
            use_context=True,
            context_filter=ContextFilter(
                collection=collection,
                artifacts=[artifact],
            ),
        ),
    )

    response = await async_test_client.post(
        "/v1/messages/validate", json=body.model_dump()
    )
    assert response.status_code == 200, response.json()

    await async_test_client.post(
        "/v1/artifacts/delete",
        json={"collection": collection, "artifact": artifact},
    )


@pytest.mark.anyio
async def test_chat_accepts_system_string(async_test_client: AsyncClient) -> None:
    """Chat route should accept `system` as a plain string."""
    body = ChatBody(
        messages=[MessageInput(content="test", role="user")],
        stream=False,
        system="You are a helpful assistant.",
        tools=tools(use_context=False),
        tool_choice=tool_choice(use_context=False),
    )

    response = await async_test_client.post("/v1/messages", json=body.model_dump())
    assert response.status_code == 200
    output = Message.model_validate(response.json())
    assert output.content is not None
    assert len(output.content) == 1
    assert isinstance(output.content[0], TextBlock)
    assert "You are a helpful assistant." in output.content[0].text


@pytest.mark.anyio
async def test_chat_accepts_system_list(async_test_client: AsyncClient) -> None:
    """Chat route should accept `system` as a list."""
    body = ChatBody(
        messages=[MessageInput(content="test list", role="user")],
        stream=False,
        system=[
            "First prompt part.",
            {
                "text": "Second prompt part.",
                "use_default_prompt": False,
                "citations": {"enabled": True},
            },
            System(text="Third prompt part.", use_default_prompt=True),
            System(
                text="Fourth prompt part.",
                use_default_prompt=False,
                extensions=[SystemExtensions.ZYLON],
            ),
        ],
        tools=tools(use_context=False),
        tool_choice=tool_choice(use_context=False),
    )

    response = await async_test_client.post("/v1/messages", json=body.model_dump())
    assert response.status_code == 200
    output = Message.model_validate(response.json())
    assert output.content is not None
    assert len(output.content) == 1
    assert isinstance(output.content[0], TextBlock)
    assert "First" in output.content[0].text
    assert "Second" in output.content[0].text
    assert "Third" in output.content[0].text
    assert "Fourth" in output.content[0].text

    # We enabled default prompt for the third part
    assistant_name = settings().chat.assistant_name
    assert assistant_name in output.content[0].text


@pytest.mark.anyio
async def test_chat_accepts_system_dict(async_test_client: AsyncClient) -> None:
    """Chat route should accept `system` as a dict."""
    body = ChatBody(
        messages=[MessageInput(content="test dict", role="user")],
        stream=False,
        system={
            "text": "Dict prompt part.",
            "use_default_prompt": False,
            "citations": {
                "enabled": True,
            },
        },
        tools=tools(use_context=False),
        tool_choice=tool_choice(use_context=False),
    )

    response = await async_test_client.post("/v1/messages", json=body.model_dump())
    assert response.status_code == 200
    output = Message.model_validate(response.json())
    assert output.content is not None
    assert len(output.content) == 1
    assert isinstance(output.content[0], TextBlock)
    assert "Dict prompt part." in output.content[0].text


@pytest.mark.anyio
async def test_chat_accepts_empty_system_list(
    async_test_client: AsyncClient,
) -> None:
    """Chat route should accept `system` as an empty list."""
    body = ChatBody(
        messages=[MessageInput(content="test empty list", role="user")],
        stream=False,
        system=[],
        tools=tools(use_context=False),
        tool_choice=tool_choice(use_context=False),
    )

    response = await async_test_client.post("/v1/messages", json=body.model_dump())
    assert response.status_code == 200
    output = Message.model_validate(response.json())
    assert output.content is not None
    assert len(output.content) == 1
    assert isinstance(output.content[0], TextBlock)


@pytest.mark.anyio
async def test_chat_cancels_llm_astream_on_client_disconnection(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    llm_started = asyncio.Event()
    llm_generator_closed = asyncio.Event()

    class SlowStreamingLLM(FunctionCallingLLMMock):
        async def astream_chat_with_tools(
            self, *args: Any, **kwargs: Any
        ) -> AsyncGenerator[ChatResponse, None]:
            async def _gen() -> AsyncGenerator[ChatResponse, None]:
                try:
                    llm_started.set()
                    yield ChatResponse(
                        message=ChatMessage(role=MessageRole.ASSISTANT, content=""),
                        delta="Hi",
                    )
                    await asyncio.sleep(30)
                    yield ChatResponse(
                        message=ChatMessage(role=MessageRole.ASSISTANT, content=""),
                        delta=" never",
                    )
                finally:
                    llm_generator_closed.set()

            return _gen()

    llm_component = injector.get(LLMComponent)
    llm_component.llm = SlowStreamingLLM()

    body = ChatBody(
        messages=[MessageInput(content="What is Python?", role="user")],
        stream=False,
    )

    request_task = asyncio.create_task(
        async_test_client.post("/v1/messages", json=body.model_dump())
    )

    try:
        await asyncio.wait_for(llm_started.wait(), timeout=5.0)
    except TimeoutError:
        request_task.cancel()
        pytest.fail("LLM astream_chat_with_tools never started")

    request_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await request_task

    try:
        await asyncio.wait_for(llm_generator_closed.wait(), timeout=5.0)
    except TimeoutError:
        pytest.fail(
            "LLM astream_chat_with_tools generator was not closed after client disconnection"
        )

    assert llm_generator_closed.is_set()


@pytest.mark.anyio
async def test_principal_is_propagated_to_background_task(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    """Principal set by the HTTP middleware must reach the asyncio task that
    processes the chat stream.

    TaskManager.create_task() uses copy_context() before calling
    asyncio.create_task(), which snapshots the ContextVar state (including
    the Principal) at the moment the task is created — i.e. while the
    middleware's principal is still active.  This test makes a request with a
    specific Authorization header and asserts that Principal.current() inside
    the spawned task returns the matching api_key.
    """
    from private_gpt.components.streaming.stream.stream_processor import StreamProcessor
    from private_gpt.server.principal import Principal

    captured: list[str | None] = []
    stream_processor = injector.get(StreamProcessor)
    original_process_stream = stream_processor.process_stream

    async def capturing_process_stream(
        correlation_id,
        stream_type,
        event_generator,
        event_handler,
        metadata=None,
        mark_completed=True,
    ):
        # This coroutine runs inside the asyncio task spawned by TaskManager.
        # The task was created with copy_context(), so Principal.current() here
        # must return the principal that was active during the HTTP request.
        captured.append(Principal.current().api_key)
        await original_process_stream(
            correlation_id,
            stream_type,
            event_generator,
            event_handler,
            metadata,
            mark_completed,
        )

    stream_processor.process_stream = capturing_process_stream  # type: ignore[method-assign]
    try:
        body = ChatBody(
            messages=[MessageInput(content="test", role="user")],
            stream=False,
        )
        response = await async_test_client.post(
            "/v1/messages",
            json=body.model_dump(),
            headers={"authorization": "Bearer sk-test-principal"},
        )
        assert response.status_code == 200
    finally:
        stream_processor.process_stream = original_process_stream  # type: ignore[method-assign]

    assert len(captured) == 1, "process_stream must be called exactly once per request"
    assert captured[0] == "sk-test-principal", (
        f"Task saw api_key={captured[0]!r}, expected 'sk-test-principal' — "
        "Principal was not propagated into the background task"
    )


@pytest.mark.anyio
async def test_concurrent_requests_dont_share_state(
    async_test_client: AsyncClient, injector: MockInjector
) -> None:
    request_1_streaming = asyncio.Event()
    request_2_can_start = asyncio.Event()
    request_1_can_finish = asyncio.Event()

    call_count = 0

    class InterleavedStreamingLLM(FunctionCallingLLMMock):
        async def astream_chat_with_tools(
            self, *args: Any, **kwargs: Any
        ) -> AsyncGenerator[ChatResponse, None]:
            async def _gen() -> AsyncGenerator[ChatResponse, None]:
                nonlocal call_count
                call_count += 1
                is_first = call_count == 1

                if is_first:
                    request_1_streaming.set()
                    # Yield one chunk, then pause to let request 2 start
                    # and trigger on_iteration_start (which clears _block_id_map)
                    yield ChatResponse(
                        message=ChatMessage(role=MessageRole.ASSISTANT, content=""),
                        delta="Hello",
                    )
                    request_2_can_start.set()
                    await request_1_can_finish.wait()
                    yield ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT, content="Hello world"
                        ),
                        delta=" world",
                    )
                else:
                    # Request 2: trigger on_iteration_start to wipe shared state
                    yield ChatResponse(
                        message=ChatMessage(role=MessageRole.ASSISTANT, content=""),
                        delta="Hi",
                    )
                    request_1_can_finish.set()
                    yield ChatResponse(
                        message=ChatMessage(
                            role=MessageRole.ASSISTANT, content="Hi there"
                        ),
                        delta=" there",
                    )

            return _gen()

    llm_component = injector.get(LLMComponent)
    llm_component.llm = InterleavedStreamingLLM()

    body = ChatBody(
        messages=[MessageInput(content="test", role="user")],
        stream=False,
    )

    async def make_request() -> int:
        response = await async_test_client.post("/v1/messages", json=body.model_dump())
        return response.status_code

    task_1 = asyncio.create_task(make_request())

    await asyncio.wait_for(request_1_streaming.wait(), timeout=5.0)
    await asyncio.wait_for(request_2_can_start.wait(), timeout=5.0)

    task_2 = asyncio.create_task(make_request())

    status_1, status_2 = await asyncio.gather(task_1, task_2)

    assert status_1 == 200, f"Request 1 failed with status {status_1}"
    assert status_2 == 200, f"Request 2 failed with status {status_2}"


# ---------------------------------------------------------------------------
# Document file tests
# ---------------------------------------------------------------------------


def _plain_doc(data: str, title: str | None = None) -> dict[str, Any]:
    """Build a document content block with a plain-text source."""
    block: dict[str, Any] = {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": data},
    }
    if title:
        block["title"] = title
    return block


def _parse_sse_tool_blocks(sse_text: str) -> tuple[list[dict], list[dict]]:
    """Return (tool_use_blocks, tool_result_blocks) found in an SSE response."""
    tool_uses: list[dict] = []
    tool_results: list[dict] = []
    for chunk in sse_text.split("\n\n"):
        if "content_block_start" not in chunk:
            continue
        data_line = next(
            (line for line in chunk.splitlines() if line.startswith("data:")), None
        )
        if not data_line:
            continue
        payload = json.loads(data_line.split("data:", 1)[1].strip())
        block = payload.get("content_block", {})
        if block.get("type") == "tool_use":
            tool_uses.append(block)
        elif block.get("type") == "tool_result":
            tool_results.append(block)
    return tool_uses, tool_results


@pytest.mark.anyio
async def test_chat_with_single_document_file(
    async_test_client: AsyncClient,
) -> None:
    """A message with one document block is processed and returns 200."""
    body = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": [
                    _plain_doc("Hello, this is a test document.", title="Test Doc"),
                    {"type": "text", "text": "Summarize the document."},
                ],
            }
        ],
        "stream": False,
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 200
    completion = Message.model_validate(response.json())
    assert completion.content is not None


@pytest.mark.anyio
async def test_chat_with_multiple_document_files_emits_one_tool_pair_per_document(
    async_test_client: AsyncClient,
) -> None:
    """3 document blocks → 3 ToolUseBlock + 3 ToolResultBlock pairs in the stream."""
    body = {
        "model": "default",
        "messages": [
            {
                "role": "user",
                "content": [
                    _plain_doc("Q1 revenue was 3.2M.", title="Q1 Report"),
                    _plain_doc("Q2 revenue was 4.1M.", title="Q2 Report"),
                    _plain_doc("Q3 revenue was 5.0M.", title="Q3 Report"),
                    {"type": "text", "text": "Compare the quarters."},
                ],
            }
        ],
        "stream": True,
    }
    response = await async_test_client.post("/v1/messages", json=body)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    tool_uses, tool_results = _parse_sse_tool_blocks(response.text)
    doc_uses = [b for b in tool_uses if b.get("name") == "document_preprocessing"]
    assert len(doc_uses) == 3, f"Expected 3 tool use blocks, got {len(doc_uses)}"
    assert (
        len(tool_results) == 3
    ), f"Expected 3 tool result blocks, got {len(tool_results)}"

    # Each result must reference a use that was emitted for this message.
    use_ids = {b["id"] for b in doc_uses}
    for result in tool_results:
        assert (
            result["tool_use_id"] in use_ids
        ), f"tool_result references unknown tool_use_id {result['tool_use_id']!r}"


@pytest.mark.anyio
async def test_chat_with_multiple_document_files_with_concurrency_limit(
    async_test_client: AsyncClient,
) -> None:
    """3 documents with max_concurrency=1 still produce 3 paired tool events."""
    original = settings().chat.preprocess.documents.max_concurrency
    settings().chat.preprocess.documents.max_concurrency = 1
    try:
        body = {
            "model": "default",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        _plain_doc("Doc A content.", title="Doc A"),
                        _plain_doc("Doc B content.", title="Doc B"),
                        _plain_doc("Doc C content.", title="Doc C"),
                        {"type": "text", "text": "What do the docs say?"},
                    ],
                }
            ],
            "stream": True,
        }
        response = await async_test_client.post("/v1/messages", json=body)
        assert response.status_code == 200

        tool_uses, tool_results = _parse_sse_tool_blocks(response.text)
        doc_uses = [b for b in tool_uses if b.get("name") == "document_preprocessing"]
        assert len(doc_uses) == 3
        assert len(tool_results) == 3
        use_ids = {b["id"] for b in doc_uses}
        for result in tool_results:
            assert result["tool_use_id"] in use_ids
    finally:
        settings().chat.preprocess.documents.max_concurrency = original
