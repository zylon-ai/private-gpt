import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, get_args
from unittest.mock import AsyncMock, Mock

import anthropic
import httpx
import pytest
from anthropic import APIError, APIStatusError
from anthropic.types import (
    MessageParam,
    ModelParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from llama_index.core.llms.llm import ToolSelection
from pytest_httpx import HTTPXMock, IteratorStream
from starlette.testclient import TestClient

from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.chat.input_models import (
    CountTokensOutput,
    ModelInfoOutput,
    ModelListOutput,
)
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.tools.tool_names import (
    INTERNAL_TOOLS,
    SEMANTIC_SEARCH_TOOL_NAME,
)
from private_gpt.events.interceptors.ping_event_interceptor import (
    _DEFAULT_PING_INTERVAL,
)
from private_gpt.events.models import TextBlock as OutTextBlock
from private_gpt.events.models import Usage
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.server.chat.chat_service import Completion as ChatCompletion
from private_gpt.server.models.models_service import ModelsService
from private_gpt.server.utils.artifact_input import ArtifactType, IngestedArtifact
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm
from tests.fixtures.mock_injector import MockInjector

WEATHER_TOOL_NAME = "get_weather"
WEATHER_TOOL_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
}


class ToolConfig:
    def __init__(
        self,
        name: str,
        input_schema: dict[str, Any] | None = None,
        context: list[ArtifactType] | None = None,
    ) -> None:
        self.name = name
        self.input_schema = input_schema
        self.context = context or []


def create_weather_tool(context: list[ArtifactType] | None = None) -> ToolConfig:
    return ToolConfig(WEATHER_TOOL_NAME, WEATHER_TOOL_SCHEMA, context)


def create_semantic_tool(context: list[ArtifactType] | None = None) -> ToolConfig:
    return ToolConfig(SEMANTIC_SEARCH_TOOL_NAME, None, context)


def generate_tool_deltas(tools: list[ToolConfig]) -> list[list[str | ToolSelection]]:
    def tool_iterator() -> Iterator[ToolSelection]:
        for tool in tools:
            if tool.name == WEATHER_TOOL_NAME:
                yield ToolSelection(
                    tool_id=WEATHER_TOOL_NAME,
                    tool_name=WEATHER_TOOL_NAME,
                    tool_kwargs={"city": "San Francisco"},
                )
            elif tool.name == SEMANTIC_SEARCH_TOOL_NAME:
                yield ToolSelection(
                    tool_id=SEMANTIC_SEARCH_TOOL_NAME,
                    tool_name=SEMANTIC_SEARCH_TOOL_NAME,
                    tool_kwargs={"query": "Lorem ipsum dolor sit amet"},
                )
            else:
                raise ValueError(f"Unknown tool: {tool.name}")

    result = [list(tool_iterator()), ["Lorem ipsum dolor sit amet"]]
    return [batch for batch in result if batch]


def convert_to_anthropic_tools(tools: list[ToolConfig]) -> list[ToolParam]:
    return [
        ToolParam(
            type=tool.name + "_v1" if tool.name in INTERNAL_TOOLS else None,
            name=tool.name,
            input_schema=tool.input_schema,
            context=[artifact.model_dump() for artifact in tool.context]
            if tool.context
            else None,
        )
        for tool in tools
    ]


def setup_mock_llm(
    injector: MockInjector,
    tools: list[ToolConfig],
    sleep_between_blocks: float = 0.0,
    sleep_between_deltas: float = 0.0,
) -> None:
    deltas = generate_tool_deltas(tools)
    mock_llm_instance = get_mock_function_calling_llm(
        deltas,
        sleep_between_blocks,
        sleep_between_deltas,
    )

    llm_component = injector.get(LLMComponent)
    llm_component.get_llm = Mock(return_value=mock_llm_instance)
    injector.bind_mock(LLMComponent, llm_component)


def create_mock_http_client(
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    is_async: bool = False,
) -> httpx.Client | httpx.AsyncClient:
    def build_response(request: httpx.Request) -> httpx.Response:
        starlette_request = test_client.build_request(
            method=request.method,
            url=request.url.path,
            headers=request.headers,
            content=request.content,
            params=request.url.params,
        )
        response = test_client.send(starlette_request)
        content_type = response.headers.get("Content-Type", "")

        if "text/event-stream" in content_type:
            raw_events = [
                (item + "\n\n").encode("utf-8")
                for item in response.text.split("\n\n")
                if item.strip()
            ]
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                stream=IteratorStream(raw_events),
            )

        if "application/json" in content_type:
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                json=response.json(),
            )

        return httpx.Response(
            status_code=response.status_code,
            headers=response.headers,
            content=response.content,
        )

    httpx_mock.add_callback(build_response)
    httpx_mock.add_response(is_reusable=True)

    return httpx.AsyncClient() if is_async else httpx.Client()


def ingest_test_artifact(test_client: TestClient) -> IngestedArtifact:
    collection_id = str(uuid.uuid4())
    artifact_id = str(uuid.uuid4())

    response = test_client.post(
        "/v1/artifacts/ingest",
        json={
            "metadata": {},
            "input": {"type": "text", "value": "Lorem ipsum dolor sit amet"},
            "collection": collection_id,
            "artifact": artifact_id,
        },
    )
    assert response.status_code == 200

    return IngestedArtifact(
        context_filter=ContextFilter(
            collection=collection_id,
            artifacts=[artifact_id],
        )
    )


def prepare_tools_with_context(
    tools: list[ToolConfig], test_client: TestClient
) -> list[ToolConfig]:
    result = []
    for tool in tools:
        if tool.name == SEMANTIC_SEARCH_TOOL_NAME and not tool.context:
            artifact = ingest_test_artifact(test_client)
            result.append(ToolConfig(tool.name, tool.input_schema, [artifact]))
        else:
            result.append(tool)
    return result


def validate_response_structure(
    response: Any,
    has_tools: bool,
    has_internal_tools: bool,
    expected_text: str | None = None,
) -> None:
    assert response.role == "assistant"
    assert len(response.content) > 0

    if has_tools:
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_use_blocks) == 1

    if has_internal_tools:
        tool_result_blocks = [b for b in response.content if b.type == "tool_result"]
        assert len(tool_result_blocks) == 1

    if expected_text is not None:
        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0].text == expected_text


def validate_streaming_response(
    collected_text: str,
    tool_use_count: int,
    tool_result_count: int,
    has_tools: bool,
    has_internal_tools: bool,
    expected_text: str | None = None,
) -> None:
    if has_tools:
        assert tool_use_count == 1

    if has_internal_tools:
        assert tool_result_count == 1

    if expected_text is not None:
        assert collected_text == expected_text


CLIENT_KWARGS = {
    "base_url": "http://testserver",
    "api_key": "test_key",
    "max_retries": 0,
}


@pytest.mark.parametrize(
    ("tools", "expected_text", "has_internal_tools"),
    [
        ([], "Lorem ipsum dolor sit amet", False),
        ([create_semantic_tool()], "Lorem ipsum dolor sit amet", True),
        ([create_weather_tool()], "Lorem ipsum dolor sit amet", False),
    ],
    ids=["normal_chat", "semantic_search_tool", "custom_weather_tool"],
)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_sync_chat_non_streaming(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    tools: list[ToolConfig],
    expected_text: str,
    has_internal_tools: bool,
) -> None:
    prepared_tools = prepare_tools_with_context(tools, test_client)
    setup_mock_llm(injector, prepared_tools)

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    response = client.messages.create(
        model="default",
        max_tokens=1024,
        messages=[MessageParam(role="user", content="Test message")],
        tools=convert_to_anthropic_tools(prepared_tools) if prepared_tools else None,
    )

    validate_response_structure(
        response,
        has_tools=bool(tools),
        has_internal_tools=has_internal_tools,
        expected_text=expected_text if not tools or has_internal_tools else None,
    )


@pytest.mark.parametrize(
    ("tools", "expected_text", "has_internal_tools"),
    [
        ([], "Lorem ipsum dolor sit amet", False),
        ([create_semantic_tool()], "Lorem ipsum dolor sit amet", True),
        ([create_weather_tool()], "Lorem ipsum dolor sit amet", False),
    ],
    ids=["normal_chat", "semantic_search_tool", "custom_weather_tool"],
)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_sync_chat_streaming(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    tools: list[ToolConfig],
    expected_text: str,
    has_internal_tools: bool,
) -> None:
    prepared_tools = prepare_tools_with_context(tools, test_client)
    setup_mock_llm(injector, prepared_tools)

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    collected_text = []
    tool_use_count = 0
    tool_result_count = 0

    with client.messages.stream(
        model="default",
        max_tokens=1024,
        messages=[MessageParam(role="user", content="Test message")],
        tools=convert_to_anthropic_tools(prepared_tools) if prepared_tools else None,
    ) as stream:
        for event in stream:
            if hasattr(event, "type"):
                if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                    collected_text.append(event.delta.text)
                elif event.type == "content_block_start" and hasattr(
                    event.content_block, "type"
                ):
                    if event.content_block.type == "tool_use":
                        tool_use_count += 1
                    elif event.content_block.type == "tool_result":
                        tool_result_count += 1

    validate_streaming_response(
        collected_text="".join(collected_text),
        tool_use_count=tool_use_count,
        tool_result_count=tool_result_count,
        has_tools=bool(tools),
        has_internal_tools=has_internal_tools,
        expected_text=expected_text if not tools or has_internal_tools else None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tools", "expected_text", "has_internal_tools"),
    [
        ([], "Lorem ipsum dolor sit amet", False),
        ([create_semantic_tool()], "Lorem ipsum dolor sit amet", True),
        ([create_weather_tool()], "Lorem ipsum dolor sit amet", False),
    ],
    ids=["normal_chat", "semantic_search_tool", "custom_weather_tool"],
)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_async_chat_non_streaming(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    tools: list[ToolConfig],
    expected_text: str,
    has_internal_tools: bool,
) -> None:
    prepared_tools = prepare_tools_with_context(tools, test_client)
    setup_mock_llm(injector, prepared_tools)

    client = anthropic.AsyncAnthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock, is_async=True)

    response = await client.messages.create(
        model="default",
        max_tokens=1024,
        messages=[MessageParam(role="user", content="Test message")],
        tools=convert_to_anthropic_tools(prepared_tools) if prepared_tools else None,
    )

    validate_response_structure(
        response,
        has_tools=bool(tools),
        has_internal_tools=has_internal_tools,
        expected_text=expected_text if not tools or has_internal_tools else None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tools", "expected_text", "has_internal_tools"),
    [
        ([], "Lorem ipsum dolor sit amet", False),
        ([create_semantic_tool()], "Lorem ipsum dolor sit amet", True),
        ([create_weather_tool()], "Lorem ipsum dolor sit amet", False),
    ],
    ids=["normal_chat", "semantic_search_tool", "custom_weather_tool"],
)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_async_chat_streaming(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    tools: list[ToolConfig],
    expected_text: str,
    has_internal_tools: bool,
) -> None:
    prepared_tools = prepare_tools_with_context(tools, test_client)
    setup_mock_llm(injector, prepared_tools)

    client = anthropic.AsyncAnthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock, is_async=True)

    collected_text = []
    tool_use_count = 0
    tool_result_count = 0

    async with client.messages.stream(
        model="default",
        max_tokens=1024,
        messages=[MessageParam(role="user", content="Test message")],
        tools=convert_to_anthropic_tools(prepared_tools) if prepared_tools else None,
    ) as stream:
        async for event in stream:
            if hasattr(event, "type"):
                if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                    collected_text.append(event.delta.text)
                elif event.type == "content_block_start" and hasattr(
                    event.content_block, "type"
                ):
                    if event.content_block.type == "tool_use":
                        tool_use_count += 1
                    elif event.content_block.type == "tool_result":
                        tool_result_count += 1

    validate_streaming_response(
        collected_text="".join(collected_text),
        tool_use_count=tool_use_count,
        tool_result_count=tool_result_count,
        has_tools=bool(tools),
        has_internal_tools=has_internal_tools,
        expected_text=expected_text if not tools or has_internal_tools else None,
    )


@pytest.mark.parametrize(
    "use_valid_context",
    [True, False],
    ids=["with_valid_context", "without_context"],
)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_semantic_search_requires_context(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    use_valid_context: bool,
) -> None:
    tool = (
        create_semantic_tool([ingest_test_artifact(test_client)])
        if use_valid_context
        else create_semantic_tool()
    )
    setup_mock_llm(injector, [tool])

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    if use_valid_context:
        response = client.messages.create(
            model="default",
            max_tokens=1024,
            messages=[MessageParam(role="user", content="Test message")],
            tools=convert_to_anthropic_tools([tool]),
        )
        assert response.role == "assistant"
    else:
        with pytest.raises(APIError):
            client.messages.create(
                model="default",
                max_tokens=1024,
                messages=[MessageParam(role="user", content="Test message")],
                tools=convert_to_anthropic_tools([tool]),
            )


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_multiple_messages_conversation(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    messages = [
        MessageParam(role="user", content="First message"),
        MessageParam(role="assistant", content="First response"),
        MessageParam(role="user", content="Second message"),
    ]

    response = client.messages.create(
        model="default",
        max_tokens=1024,
        messages=messages,
    )

    assert response.role == "assistant"
    assert len(response.content) > 0


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_legacy_completion_endpoint(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    chat_service = Mock()
    chat_service.chat = AsyncMock(
        return_value=ChatCompletion(
            content=[OutTextBlock(text="Hello!")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=1, output_tokens=1),
        )
    )
    injector.bind_mock(ChatService, chat_service)

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    response = client.completions.create(
        model="default",
        max_tokens_to_sample=64,
        prompt="\n\nHuman: Say hello\n\nAssistant:",
    )

    assert response.type == "completion"
    assert isinstance(response.completion, str)


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_legacy_completion_endpoint_forwards_all_supported_params(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    chat_service = Mock()
    chat_service.chat = AsyncMock(
        return_value=ChatCompletion(
            content=[OutTextBlock(text="Hello from completion")],
            stop_reason="end_turn",
            usage=Usage(input_tokens=1, output_tokens=1),
        )
    )
    injector.bind_mock(ChatService, chat_service)

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    _ = client.completions.create(
        model="default",
        max_tokens_to_sample=128,
        prompt="\n\nHuman: Test forwarding\n\nAssistant:",
        metadata={"user_id": "sdk_user"},
        stop_sequences=["STOP", "END"],
        stream=False,
        temperature=0.42,
        top_k=25,
        top_p=0.91,
    )

    chat_service.chat.assert_awaited_once()
    chat_request = chat_service.chat.await_args.args[0]
    assert chat_request.system.model is None
    assert chat_request.stream is False
    assert chat_request.sampling_params["max_tokens"] == 128
    assert chat_request.sampling_params["temperature"] == 0.42
    assert chat_request.sampling_params["top_k"] == 25
    assert chat_request.sampling_params["top_p"] == 0.91
    assert len(chat_request.messages) == 1
    assert chat_request.messages[0].role == "user"
    assert chat_request.messages[0].content == "Test forwarding"


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_models_list_and_retrieve_endpoints(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    mock_model = ModelInfoOutput(
        id="mock",
        created_at=datetime(1970, 1, 1, tzinfo=UTC),
        display_name="Mock Model",
        type="model",
        max_tokens=1024,
        max_input_tokens=4096,
        capabilities=None,
    )
    models_service = Mock()
    models_service.list_models = Mock(
        return_value=ModelListOutput(
            data=[mock_model],
            first_id=mock_model.id,
            last_id=mock_model.id,
            has_more=False,
        )
    )
    models_service.get_model = Mock(return_value=mock_model)
    injector.bind_mock(ModelsService, models_service)

    # NOTE: create_mock_http_client currently registers a one-shot callback.
    # Use separate clients/mocks for list and retrieve so each call gets routed.
    list_client = anthropic.Anthropic(**CLIENT_KWARGS)
    list_client._client = create_mock_http_client(test_client, httpx_mock)
    listed = list_client.models.list()
    assert listed.data

    httpx_mock.reset()
    retrieve_client = anthropic.Anthropic(**CLIENT_KWARGS)
    retrieve_client._client = create_mock_http_client(test_client, httpx_mock)
    first_id = listed.data[0].id
    retrieved = retrieve_client.models.retrieve(first_id)
    assert retrieved.id == first_id
    assert retrieved.type == "model"


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_models_list_endpoint_forwards_pagination_params(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    mock_model = ModelInfoOutput(
        id="mock-after",
        created_at=datetime(1970, 1, 1, tzinfo=UTC),
        display_name="Mock After",
        type="model",
        max_tokens=2048,
        max_input_tokens=8192,
        capabilities=None,
    )
    models_service = Mock()
    models_service.list_models = Mock(
        return_value=ModelListOutput(
            data=[mock_model],
            first_id=mock_model.id,
            last_id=mock_model.id,
            has_more=False,
        )
    )
    injector.bind_mock(ModelsService, models_service)

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    listed = client.models.list(after_id="model_1", limit=7)
    assert listed.data[0].id == "mock-after"

    models_service.list_models.assert_called_once_with(None, "model_1", 7)


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_messages_count_tokens_endpoint(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    chat_service = Mock()
    chat_service.count_tokens = AsyncMock(
        return_value=CountTokensOutput(input_tokens=3)
    )
    injector.bind_mock(ChatService, chat_service)

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    counted = client.messages.count_tokens(
        model="default",
        messages=[MessageParam(role="user", content="Hello, world!")],
    )
    assert counted.input_tokens > 0


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_messages_count_tokens_endpoint_forwards_message_input_and_options(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    injector.get(ChatService)
    chat_service = Mock()
    chat_service.count_tokens = AsyncMock(
        return_value=CountTokensOutput(input_tokens=11)
    )
    injector.bind_mock(ChatService, chat_service)

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    counted = client.messages.count_tokens(
        model="default",
        messages=[
            MessageParam(role="user", content="First user message"),
            MessageParam(role="assistant", content="Assistant reply"),
            MessageParam(role="user", content="Second user message"),
        ],
        system="System guidance",
        tools=[
            {
                "name": "get_weather",
                "description": "Get weather by city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        tool_choice={"type": "auto"},
    )
    assert counted.input_tokens == 11

    chat_service.count_tokens.assert_awaited_once()
    chat_request = chat_service.count_tokens.await_args.args[0]
    assert chat_request.system.model is None
    assert len(chat_request.messages) == 3
    assert chat_request.messages[0].role == "user"
    assert chat_request.messages[1].role == "assistant"
    assert chat_request.messages[2].role == "user"
    assert chat_request.system.prompt == "System guidance"
    assert len(chat_request.tool_config.tools) == 1
    assert chat_request.tool_config.tools[0].name == "get_weather"


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_messages_count_tokens_forwards_output_config_effort_and_format(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    chat_service = Mock()
    chat_service.count_tokens = AsyncMock(
        return_value=CountTokensOutput(input_tokens=21)
    )
    injector.bind_mock(ChatService, chat_service)

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    counted = client.messages.count_tokens(
        model="default",
        messages=[MessageParam(role="user", content="Return a typed object")],
        output_config={
            "effort": "max",
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        },
    )
    assert counted.input_tokens == 21

    chat_service.count_tokens.assert_awaited_once()
    chat_request = chat_service.count_tokens.await_args.args[0]
    assert chat_request.thinking.enabled is True
    assert chat_request.thinking.type == "max"
    assert chat_request.response_format is not None
    assert chat_request.response_format.output_cls is not None


@pytest.mark.asyncio
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_async_messages_count_tokens_endpoint_forwards_message_input(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    chat_service = Mock()
    chat_service.count_tokens = AsyncMock(
        return_value=CountTokensOutput(input_tokens=9)
    )
    injector.bind_mock(ChatService, chat_service)

    client = anthropic.AsyncAnthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock, is_async=True)

    counted = await client.messages.count_tokens(
        model="default",
        messages=[MessageParam(role="user", content="Async count tokens input")],
        system="Async system",
    )
    assert counted.input_tokens == 9

    chat_service.count_tokens.assert_awaited_once()
    chat_request = chat_service.count_tokens.await_args.args[0]
    assert chat_request.system.model is None
    assert len(chat_request.messages) == 1
    assert chat_request.messages[0].content == "Async count tokens input"
    assert chat_request.system.prompt == "Async system"


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_empty_tool_list_behaves_like_no_tools(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    response_with_none = client.messages.create(
        model="default",
        max_tokens=1024,
        messages=[MessageParam(role="user", content="Test")],
        tools=None,
    )

    assert response_with_none, "Responses should not be None"
    assert response_with_none.role == "assistant"


@pytest.mark.asyncio
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_async_empty_messages_raises_error(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    client = anthropic.AsyncAnthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock, is_async=True)

    with pytest.raises(APIError):
        await client.messages.create(
            model="default",
            max_tokens=1024,
            messages=[],
        )


@pytest.mark.parametrize(
    "extra_params",
    [
        {"temperature": 0.7},
        {"top_p": 0.9},
        {"top_k": 40},
        {"temperature": 0.5, "top_p": 0.95, "top_k": 50},
        {"stop_sequences": ["STOP", "END"]},
        {"metadata": {"user_id": "test_user"}},
    ],
    ids=[
        "temperature",
        "top_p",
        "top_k",
        "multiple_params",
        "stop_sequences",
        "metadata",
    ],
)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_additional_request_parameters(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    extra_params: dict[str, Any],
) -> None:
    setup_mock_llm(injector, [])

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    response = client.messages.create(
        model="default",
        max_tokens=1024,
        messages=[MessageParam(role="user", content="Test message")],
        **extra_params,
    )

    assert response, "Responses should not be None"
    assert response.role == "assistant"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_params",
    [
        {"temperature": 0.7},
        {"top_p": 0.9, "top_k": 40},
        {"stop_sequences": ["STOP"]},
    ],
    ids=["temperature", "sampling_params", "stop_sequences"],
)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_async_additional_request_parameters(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    extra_params: dict[str, Any],
) -> None:
    setup_mock_llm(injector, [])

    client = anthropic.AsyncAnthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock, is_async=True)

    response = await client.messages.create(
        model="default",
        max_tokens=1024,
        messages=[MessageParam(role="user", content="Test message")],
        **extra_params,
    )

    assert response, "Responses should not be None"
    assert response.role == "assistant"


@pytest.mark.parametrize(
    "max_tokens_value",
    [1, 100, 1024, 4096, 8192],
    ids=["min", "small", "medium", "large", "xlarge"],
)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_various_max_tokens_values(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    max_tokens_value: int,
) -> None:
    setup_mock_llm(injector, [])

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    response = client.messages.create(
        model="default",
        max_tokens=max_tokens_value,
        messages=[MessageParam(role="user", content="Test message")],
    )

    assert response, "Responses should not be None"
    assert response.role == "assistant"


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_system_message_parameter(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    response = client.messages.create(
        model="default",
        max_tokens=1024,
        system="You are a helpful assistant.",
        messages=[MessageParam(role="user", content="Test message")],
    )

    assert response, "Responses should not be None"
    assert response.role == "assistant"


@pytest.mark.asyncio
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_async_system_message_parameter(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    client = anthropic.AsyncAnthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock, is_async=True)

    response = await client.messages.create(
        model="default",
        max_tokens=1024,
        system="You are a helpful assistant.",
        messages=[MessageParam(role="user", content="Test message")],
    )

    assert response.role == "assistant"


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_invalid_tool_definition_raises_error(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    with pytest.raises(APIError):
        client.messages.create(
            model="default",
            max_tokens=1024,
            messages=[MessageParam(role="user", content="Test message")],
            tools=[{"invalid": "tool"}],
        )


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_message_with_only_assistant_role_raises_error(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    with pytest.raises(APIError):
        client.messages.create(
            model="default",
            max_tokens=1024,
            messages=[MessageParam(role="assistant", content="I can help!")],
        )


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_streaming_with_additional_parameters(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    with (
        client.messages.stream(
            model="custom-model",
            max_tokens=2048,
            temperature=0.8,
            top_p=0.95,
            messages=[MessageParam(role="user", content="Test message")],
        ) as stream,
        pytest.raises(APIStatusError),
    ):
        list(stream)


@pytest.mark.asyncio
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_async_streaming_with_additional_parameters(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    client = anthropic.AsyncAnthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock, is_async=True)

    async with client.messages.stream(
        model="custom-model",
        max_tokens=2048,
        temperature=0.8,
        top_p=0.95,
        messages=[MessageParam(role="user", content="Test message")],
    ) as stream:

        async def collector() -> None:
            try:
                async for _ in stream:
                    continue
            except APIStatusError:
                raise

        with pytest.raises(APIStatusError):
            await collector()


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        pytest.param(400, "invalid_request_error", id="invalid_request_error"),
        pytest.param(413, "request_too_large", id="request_too_large"),
        pytest.param(500, "api_error", id="api_error"),
    ],
)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_http_error_parsing(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    status_code: int,
    error_type: str,
) -> None:
    """Test that HTTP errors are correctly parsed and raise appropriate exceptions."""
    setup_mock_llm(injector, [])

    def error_callback(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=status_code,
            json={
                "type": "error",
                "error": {
                    "type": error_type,
                    "message": f"Test error for {error_type}",
                },
            },
        )

    httpx_mock.add_callback(error_callback)

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = httpx.Client()

    with pytest.raises(APIStatusError) as exc_info:
        client.messages.create(
            model="default",
            max_tokens=1024,
            messages=[MessageParam(role="user", content="Test message")],
        )

    assert exc_info.value.status_code == status_code
    assert error_type in str(exc_info.value)


@pytest.mark.parametrize(
    ("error_type", "max_tokens", "message_content"),
    [
        pytest.param(
            "invalid_request_error",
            1024,
            "Test message",
            id="invalid_request_error",
        ),
        pytest.param(
            "request_too_large",
            1024,
            "x" * 50_000,
            id="request_too_large",
        ),
        pytest.param(
            "api_error",
            1024,
            "Test message",
            id="api_error",
        ),
    ],
)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_http_error_parsing_streaming(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    error_type: str,
    max_tokens: int,
    message_content: str,
) -> None:
    """Test that HTTP errors are correctly parsed in streaming mode via SSE events."""
    setup_mock_llm(injector, [])

    def error_sse_callback(request: httpx.Request) -> httpx.Response:
        error_event = f'event: error\ndata: {{"type": "error", "error": {{"type": "{error_type}", "message": "Test error for {error_type}"}}}}\n\n'
        return httpx.Response(
            status_code=200,  # SSE always returns 200, errors are in the stream
            headers={"Content-Type": "text/event-stream"},
            content=error_event.encode("utf-8"),
        )

    httpx_mock.add_callback(error_sse_callback)

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = httpx.Client()

    with pytest.raises(anthropic.APIStatusError) as exc_info:  # noqa: SIM117, PT012
        with client.messages.stream(
            model="default",
            max_tokens=max_tokens,
            messages=[MessageParam(role="user", content=message_content)],
        ) as stream:
            for _ in stream:
                pass

    assert exc_info.value.status_code == 200
    assert error_type in str(exc_info.value)


@pytest.mark.parametrize(
    ("status_code", "error_type", "max_tokens", "message_content"),
    [
        pytest.param(
            413,
            "request_too_large",
            1024,
            "x" * 50_000,
            id="request_too_large",
        ),
    ],
)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_real_http_error_parsing(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    status_code: int,
    error_type: str,
    max_tokens: int,
    message_content: str,
) -> None:
    """Test that HTTP errors are correctly parsed and raise appropriate exceptions."""
    setup_mock_llm(injector, [])

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    with pytest.raises(anthropic.APIStatusError) as exc_info:
        client.messages.create(
            model="default",
            max_tokens=max_tokens,
            messages=[MessageParam(role="user", content=message_content)],
        )

    assert exc_info.value.status_code == status_code
    assert error_type in str(exc_info.value)


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_http_error_parsing_streaming_real(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    client = anthropic.Anthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock)

    with pytest.raises(anthropic.APIStatusError) as exc_info:  # noqa: SIM117, PT012
        with client.messages.stream(
            model="default",
            max_tokens=1024,
            messages=[MessageParam(role="user", content="x" * 50_000)],
        ) as stream:
            for _ in stream:
                pass

    assert exc_info.value.status_code == 200
    assert "request_too_large" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_custom_content_blocks(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    client = anthropic.AsyncAnthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock, is_async=True)

    messages = [
        MessageParam(role="user", content="This is message number"),
        MessageParam(
            role="assistant",
            content=[
                ToolUseBlockParam(
                    id="tool-use-123",
                    input={"query": "Lorem ipsum dolor sit amet"},
                    name="semantic_search",
                    type="tool_use",
                )
            ],
        ),
        MessageParam(
            role="user",
            content=[
                ToolResultBlockParam(
                    tool_use_id="tool-use-123",
                    type="tool_result",
                    content="Tool result content" * 50000,
                )
            ],
        ),
        MessageParam(role="assistant", content="Processing complete."),
        MessageParam(role="user", content="Continue the response."),
    ]

    response = await client.messages.create(
        model=None,
        max_tokens=1024,
        messages=messages,
    )

    validate_response_structure(
        response,
        has_tools=False,
        has_internal_tools=False,
        expected_text="Lorem ipsum dolor sit amet",
    )


@pytest.mark.asyncio
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_streaming_ping_events_with_slow_response(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    tools = [create_semantic_tool([ingest_test_artifact(test_client)])]
    setup_mock_llm(injector, tools, _DEFAULT_PING_INTERVAL + 1)

    client = anthropic.AsyncAnthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock, is_async=True)

    async with client.messages.stream(
        model="claude-opus-4-6",  # default
        max_tokens=1024,
        messages=[MessageParam(role="user", content="Test message")],
        tools=convert_to_anthropic_tools(tools),
    ) as stream:
        async for _ in stream:
            continue


ALL_CLAUDE_MODELS = list(get_args(get_args(ModelParam)[0]))


@pytest.mark.parametrize("model", ALL_CLAUDE_MODELS, ids=ALL_CLAUDE_MODELS)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_all_models_run_without_crash(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    model: str,
) -> None:
    setup_mock_llm(injector, [])

    client = anthropic.AsyncAnthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock, is_async=True)

    response = await client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[MessageParam(role="user", content="Test message")],
    )

    assert response.role == "assistant"
    assert len(response.content) > 0


_BASE = {
    "model": "default",
    "max_tokens": 1024,
    "messages": [MessageParam(role="user", content="Test message")],
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra_params",
    [
        pytest.param({}, id="baseline"),
        pytest.param({"system": "You are a helpful assistant."}, id="system_str"),
        pytest.param(
            {"system": [{"type": "text", "text": "You are helpful."}]},
            id="system_blocks",
        ),
        pytest.param(
            {
                "system": [
                    {"type": "text", "text": "You are helpful."},
                    {"type": "text", "text": "Follow the user instructions."},
                ]
            }
        ),
        pytest.param({"temperature": 0.0}, id="temperature_min"),
        pytest.param({"temperature": 1.0}, id="temperature_max"),
        pytest.param({"top_p": 0.9}, id="top_p"),
        pytest.param({"top_k": 40}, id="top_k"),
        pytest.param({"temperature": 0.5, "top_p": 0.95}, id="temperature_and_top_p"),
        pytest.param({"temperature": 0.5, "top_k": 50}, id="temperature_and_top_k"),
        pytest.param({"stop_sequences": ["STOP"]}, id="stop_sequences_single"),
        pytest.param(
            {"stop_sequences": ["STOP", "END", "DONE"]}, id="stop_sequences_multiple"
        ),
        pytest.param({"metadata": {"user_id": "user_123"}}, id="metadata"),
        pytest.param({"service_tier": "auto"}, id="service_tier_auto"),
        pytest.param(
            {"service_tier": "standard_only"}, id="service_tier_standard_only"
        ),
        pytest.param({"inference_geo": "us"}, id="inference_geo"),
        pytest.param({"stream": False}, id="stream_false_explicit"),
        pytest.param(
            {
                "system": "You are a helpful assistant.",
                "temperature": 0.7,
                "top_p": 0.9,
                "stop_sequences": ["STOP"],
                "metadata": {"user_id": "test"},
                "service_tier": "auto",
            },
            id="all_scalar_params",
        ),
        pytest.param(
            {
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a city.",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
            },
            id="tools_only",
        ),
        pytest.param(
            {
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a city.",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
                "tool_choice": {"type": "auto"},
            },
            id="tools_with_tool_choice_auto",
        ),
        pytest.param(
            {
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a city.",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
                "tool_choice": {"type": "any"},
            },
            id="tools_with_tool_choice_any",
        ),
        pytest.param(
            {
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather for a city.",
                        "input_schema": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    }
                ],
                "tool_choice": {"type": "tool", "name": "get_weather"},
            },
            id="tools_with_tool_choice_specific",
        ),
        pytest.param(
            {"tool_choice": {"type": "none"}},
            id="tool_choice_none",
        ),
        pytest.param(
            {
                "messages": [
                    MessageParam(role="user", content="Hello"),
                    MessageParam(role="assistant", content="Hi!"),
                    MessageParam(role="user", content="Follow-up"),
                ]
            },
            id="multi_turn_messages",
        ),
        pytest.param({"max_tokens": 1}, id="max_tokens_min"),
        pytest.param({"max_tokens": 8192}, id="max_tokens_large"),
    ],
)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_all_create_parameter_combinations(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    extra_params: dict[str, Any],
) -> None:
    tools = (
        [create_weather_tool()]
        if extra_params.get("tools")
        and extra_params.get("tool_choice", {}).get("type") in ("any", "tool")
        else []
    )
    setup_mock_llm(injector, tools)

    client = anthropic.AsyncAnthropic(**CLIENT_KWARGS)
    client._client = create_mock_http_client(test_client, httpx_mock, is_async=True)

    params = {**_BASE, **extra_params}
    response = await client.messages.create(**params)

    assert response.role == "assistant"
    assert len(response.content) > 0
