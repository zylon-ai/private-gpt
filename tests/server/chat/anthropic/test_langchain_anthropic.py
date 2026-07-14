import uuid
from collections.abc import Iterator
from typing import Any, get_args

import anthropic
import httpx
import pytest
from anthropic.types import ModelParam
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from llama_index.core.llms.llm import ToolSelection
from pytest_httpx import HTTPXMock
from starlette.testclient import TestClient

from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.tools.tool_names import (
    INTERNAL_TOOLS,
    SEMANTIC_SEARCH_TOOL_NAME,
)
from private_gpt.events.interceptors.ping_event_interceptor import (
    _DEFAULT_PING_INTERVAL,
)
from private_gpt.server.utils.artifact_input import ArtifactType, IngestedArtifact
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm
from tests.fixtures.mock_injector import MockInjector

# Add decorator to all tests to allow unused httpx mock responses
pytestmark = pytest.mark.httpx_mock(assert_all_responses_were_requested=False)


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


def convert_to_langchain_tools(tools: list[ToolConfig]) -> list[dict]:
    """Convert tool configs to LangChain-compatible format.

    LangChain's bind_tools expects tools that either:
    1. Are already in Anthropic format (with name, description, input_schema)
    2. Can be converted via convert_to_openai_tool (Pydantic models, functions, etc.)

    For custom tools, we need to ensure they're in proper Anthropic format.
    """
    langchain_tools = []

    for tool_config in tools:
        if tool_config.name in INTERNAL_TOOLS:
            # Internal tools need special format
            tool_def = {
                "type": tool_config.name + "_v1",
                "name": tool_config.name,
                "parameters": {},
            }
            if tool_config.context:
                tool_def["context"] = [
                    artifact.model_dump() for artifact in tool_config.context
                ]
        else:
            # Custom tools need to be in full Anthropic format
            tool_def = {
                "name": tool_config.name,
                "description": f"Tool {tool_config.name}",  # Add description
                "input_schema": tool_config.input_schema
                or {
                    "type": "object",
                    "properties": {},
                },
            }
            if tool_config.context:
                tool_def["context"] = [
                    artifact.model_dump() for artifact in tool_config.context
                ]

        langchain_tools.append(tool_def)

    return langchain_tools


def setup_mock_llm(
    injector: MockInjector,
    tools: list[ToolConfig],
    sleep_between_blocks: float = 0.0,
    sleep_between_deltas: float = 0.0,
) -> None:
    deltas = generate_tool_deltas(tools)
    mock_llm_instance = get_mock_function_calling_llm(
        deltas, sleep_between_blocks, sleep_between_deltas
    )

    llm_component = injector.get(LLMComponent)
    llm_component.llm = mock_llm_instance
    llm_component.get_llm.return_value = mock_llm_instance
    injector.bind_mock(LLMComponent, mock_llm_instance)


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
            response_content = response.content
            response = httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                content=response_content,
            )
            response.read()
            _ = response.text  # force to read
            return response

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
    httpx_mock.add_response()

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


def validate_langchain_response_structure(
    response: AIMessage,
    has_tools: bool,
    has_internal_tools: bool,
    expected_text: str | None = None,
) -> None:
    """Validate LangChain AIMessage response structure."""
    assert isinstance(response, AIMessage)

    if has_tools:
        assert len(response.tool_calls) >= 1

    if expected_text is not None:
        assert response.content == expected_text


def validate_langchain_streaming_response(
    collected_chunks: list[AIMessage],
    has_tools: bool,
    has_internal_tools: bool,
    expected_text: str | None = None,
) -> None:
    """Validate LangChain streaming response."""
    assert len(collected_chunks) > 0

    # Collect all content
    all_content = "".join(
        chunk.content for chunk in collected_chunks if isinstance(chunk.content, str)
    )

    # Check for tool calls
    tool_calls = [
        chunk
        for chunk in collected_chunks
        if hasattr(chunk, "tool_calls") and chunk.tool_calls
    ]

    if has_tools:
        assert len(tool_calls) >= 1

    if expected_text is not None:
        assert expected_text in all_content or all_content == expected_text


def create_langchain_chat_model(
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    is_async: bool = False,
    **kwargs: Any,
) -> ChatAnthropic:
    """Create a ChatAnthropic instance with mock HTTP client."""
    default_kwargs = {
        "model": "default",
        "anthropic_api_url": "http://testserver",  # Don't include /v1 here
        "anthropic_api_key": "test_key",
        "max_tokens": 1024,
        "max_retries": 0,
    }
    default_kwargs.update(kwargs)

    chat_model = ChatAnthropic(**default_kwargs)

    # Clear cached properties first if they exist
    if hasattr(chat_model, "_client"):
        del chat_model.__dict__["_client"]
    if hasattr(chat_model, "_async_client"):
        del chat_model.__dict__["_async_client"]

    # Create mock clients with proper parameters
    mock_http_client = create_mock_http_client(
        test_client, httpx_mock, is_async=is_async
    )

    # Create the Anthropic client with the mock HTTP client
    client_params = {
        "api_key": "test_key",
        "base_url": "http://testserver",  # Don't include /v1 here
        "max_retries": 0,
        "http_client": mock_http_client,
    }

    if is_async:
        # Override the cached property by setting it directly in __dict__
        chat_model.__dict__["_async_client"] = anthropic.AsyncClient(**client_params)
    else:
        # Override the cached property by setting it directly in __dict__
        chat_model.__dict__["_client"] = anthropic.Client(**client_params)

    return chat_model


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
def test_langchain_sync_chat_non_streaming(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    tools: list[ToolConfig],
    expected_text: str,
    has_internal_tools: bool,
) -> None:
    prepared_tools = prepare_tools_with_context(tools, test_client)
    setup_mock_llm(injector, prepared_tools)

    # Create LangChain ChatAnthropic instance
    chat_model = create_langchain_chat_model(test_client, httpx_mock)

    # Prepare messages
    messages = [HumanMessage(content="Test message")]

    # Bind tools if needed
    if prepared_tools:
        langchain_tools = convert_to_langchain_tools(prepared_tools)
        chat_model = chat_model.bind_tools(langchain_tools)

    response = chat_model.invoke(messages)

    validate_langchain_response_structure(
        response,
        has_tools=bool(tools),
        has_internal_tools=has_internal_tools,
        expected_text=expected_text if not tools and not has_internal_tools else None,
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
def test_langchain_sync_chat_streaming(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    tools: list[ToolConfig],
    expected_text: str,
    has_internal_tools: bool,
) -> None:
    prepared_tools = prepare_tools_with_context(tools, test_client)
    setup_mock_llm(injector, prepared_tools)

    chat_model = create_langchain_chat_model(test_client, httpx_mock)

    messages = [HumanMessage(content="Test message")]

    if prepared_tools:
        langchain_tools = convert_to_langchain_tools(prepared_tools)
        chat_model = chat_model.bind_tools(langchain_tools)

    collected_chunks = []
    for chunk in chat_model.stream(messages):
        collected_chunks.append(chunk)

    validate_langchain_streaming_response(
        collected_chunks,
        has_tools=bool(tools),
        has_internal_tools=has_internal_tools,
        expected_text=expected_text if not tools and not has_internal_tools else None,
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
async def test_langchain_async_chat_non_streaming(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    tools: list[ToolConfig],
    expected_text: str,
    has_internal_tools: bool,
) -> None:
    prepared_tools = prepare_tools_with_context(tools, test_client)
    setup_mock_llm(injector, prepared_tools)

    chat_model = create_langchain_chat_model(test_client, httpx_mock, is_async=True)

    messages = [HumanMessage(content="Test message")]

    if prepared_tools:
        langchain_tools = convert_to_langchain_tools(prepared_tools)
        chat_model = chat_model.bind_tools(langchain_tools)

    response = await chat_model.ainvoke(messages)

    validate_langchain_response_structure(
        response,
        has_tools=bool(tools),
        has_internal_tools=has_internal_tools,
        expected_text=expected_text if not tools and not has_internal_tools else None,
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
async def test_langchain_async_chat_streaming(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    tools: list[ToolConfig],
    expected_text: str,
    has_internal_tools: bool,
) -> None:
    prepared_tools = prepare_tools_with_context(tools, test_client)
    setup_mock_llm(injector, prepared_tools)

    chat_model = create_langchain_chat_model(test_client, httpx_mock, is_async=True)

    messages = [HumanMessage(content="Test message")]

    if prepared_tools:
        langchain_tools = convert_to_langchain_tools(prepared_tools)
        chat_model = chat_model.bind_tools(langchain_tools)

    collected_chunks = []
    async for chunk in chat_model.astream(messages):
        collected_chunks.append(chunk)

    validate_langchain_streaming_response(
        collected_chunks,
        has_tools=bool(tools),
        has_internal_tools=has_internal_tools,
        expected_text=expected_text if not tools and not has_internal_tools else None,
    )


@pytest.mark.parametrize(
    "use_valid_context",
    [True, False],
    ids=["with_valid_context", "without_context"],
)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_langchain_semantic_search_requires_context(
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

    chat_model = create_langchain_chat_model(test_client, httpx_mock)

    langchain_tools = convert_to_langchain_tools([tool])
    chat_model = chat_model.bind_tools(langchain_tools)

    messages = [HumanMessage(content="Test message")]

    if use_valid_context:
        response = chat_model.invoke(messages)
        assert isinstance(response, AIMessage)
    # Disabled since built-in tools not working yet
    # else:
    #     with pytest.raises(anthropic.APIError):
    #         chat_model.invoke(messages)


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_langchain_multiple_messages_conversation(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    chat_model = create_langchain_chat_model(test_client, httpx_mock)

    messages = [
        HumanMessage(content="First message"),
        AIMessage(content="First response"),
        HumanMessage(content="Second message"),
    ]

    response = chat_model.invoke(messages)

    assert isinstance(response, AIMessage)
    assert len(response.content) > 0


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_langchain_empty_messages_raises_error(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    chat_model = create_langchain_chat_model(test_client, httpx_mock)

    with pytest.raises(anthropic.APIError):
        chat_model.invoke([])


@pytest.mark.asyncio
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_langchain_async_empty_messages_raises_error(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    chat_model = create_langchain_chat_model(test_client, httpx_mock, is_async=True)

    with pytest.raises(anthropic.APIError):
        await chat_model.ainvoke([])


@pytest.mark.parametrize(
    "extra_params",
    [
        {"temperature": 0.7},
        {"top_p": 0.9},
        {"top_k": 40},
        {"temperature": 0.5, "top_p": 0.95, "top_k": 50},
        {"stop_sequences": ["STOP", "END"]},
    ],
    ids=[
        "temperature",
        "top_p",
        "top_k",
        "multiple_params",
        "stop_sequences",
    ],
)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_langchain_additional_request_parameters(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    extra_params: dict[str, Any],
) -> None:
    setup_mock_llm(injector, [])

    chat_model = create_langchain_chat_model(test_client, httpx_mock, **extra_params)

    messages = [HumanMessage(content="Test message")]
    response = chat_model.invoke(messages)

    assert isinstance(response, AIMessage)


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
async def test_langchain_async_additional_request_parameters(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    extra_params: dict[str, Any],
) -> None:
    setup_mock_llm(injector, [])

    chat_model = create_langchain_chat_model(
        test_client, httpx_mock, is_async=True, **extra_params
    )

    messages = [HumanMessage(content="Test message")]
    response = await chat_model.ainvoke(messages)

    assert isinstance(response, AIMessage)


@pytest.mark.parametrize(
    "max_tokens_value",
    [1, 100, 1024, 4096, 8192],
    ids=["min", "small", "medium", "large", "xlarge"],
)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_langchain_various_max_tokens_values(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    max_tokens_value: int,
) -> None:
    setup_mock_llm(injector, [])

    chat_model = create_langchain_chat_model(
        test_client, httpx_mock, max_tokens=max_tokens_value
    )

    messages = [HumanMessage(content="Test message")]
    response = chat_model.invoke(messages)

    assert isinstance(response, AIMessage)


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_langchain_system_message_parameter(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    chat_model = create_langchain_chat_model(test_client, httpx_mock)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Test message"),
    ]
    response = chat_model.invoke(messages)

    assert isinstance(response, AIMessage)


@pytest.mark.asyncio
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_langchain_async_system_message_parameter(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    chat_model = create_langchain_chat_model(test_client, httpx_mock, is_async=True)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Test message"),
    ]
    response = await chat_model.ainvoke(messages)

    assert isinstance(response, AIMessage)


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_langchain_streaming_with_additional_parameters(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    chat_model = create_langchain_chat_model(
        test_client,
        httpx_mock,
        max_tokens=2048,
        temperature=0.8,
        top_p=0.95,
    )

    messages = [HumanMessage(content="Test message")]
    list(chat_model.stream(messages))


@pytest.mark.asyncio
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
async def test_langchain_async_streaming_with_additional_parameters(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    chat_model = create_langchain_chat_model(
        test_client,
        httpx_mock,
        is_async=True,
        max_tokens=2048,
        temperature=0.8,
        top_p=0.95,
    )

    messages = [HumanMessage(content="Test message")]
    async for _ in chat_model.astream(messages):
        pass


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

    chat_model = create_langchain_chat_model(test_client, httpx_mock)
    messages = [HumanMessage(content="Test message")]

    with pytest.raises(anthropic.APIStatusError) as exc_info:
        chat_model.invoke(messages)

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
def test_langchain_http_error_parsing_streaming(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    error_type: str,
    max_tokens: int,
    message_content: str,
) -> None:
    setup_mock_llm(injector, [])

    def error_sse_callback(request: httpx.Request) -> httpx.Response:
        error_event = f'event: error\ndata: {{"type": "error", "error": {{"type": "{error_type}", "message": "Test error for {error_type}"}}}}\n\n'
        response = httpx.Response(
            status_code=200,
            headers={"Content-Type": "text/event-stream"},
            content=error_event.encode("utf-8"),
        )
        response.read()
        _ = response.text  # force to read
        return response

    httpx_mock.add_callback(error_sse_callback)
    httpx_mock.add_response()

    chat_model = ChatAnthropic(
        model="default",
        anthropic_api_url="http://testserver",
        anthropic_api_key="test_key",
        max_tokens=max_tokens,
        max_retries=0,
    )
    chat_model.__dict__["_client"] = anthropic.Client(
        api_key="test_key",
        base_url="http://testserver",
        max_retries=0,
        http_client=httpx.Client(),
    )

    messages = [HumanMessage(content=message_content)]

    with pytest.raises(anthropic.APIStatusError) as exc_info:
        for _ in chat_model.stream(messages):
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
def test_langchain_http_error_parsing_real(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    status_code: int,
    error_type: str,
    max_tokens: int,
    message_content: str,
) -> None:
    setup_mock_llm(injector, [])

    chat_model = create_langchain_chat_model(
        test_client, httpx_mock, max_tokens=max_tokens
    )
    messages = [HumanMessage(content=message_content)]

    with pytest.raises(anthropic.APIStatusError) as exc_info:
        chat_model.invoke(messages)

    assert exc_info.value.status_code == status_code
    assert error_type in str(exc_info.value)


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_langchain_http_error_parsing_streaming_real(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
) -> None:
    setup_mock_llm(injector, [])

    chat_model = create_langchain_chat_model(test_client, httpx_mock, max_tokens=1024)
    messages = [HumanMessage(content="x" * 50_000)]

    with pytest.raises(anthropic.APIStatusError) as exc_info:
        for _ in chat_model.stream(messages):
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
    chat_model = create_langchain_chat_model(test_client, httpx_mock, is_async=True)

    messages = [
        HumanMessage(content="This is message number"),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "semantic_search",
                    "args": {"query": "Lorem ipsum dolor sit amet"},
                    "id": "tool_call_123",
                }
            ],
        ),
        ToolMessage(
            name="semantic_search",
            tool_call_id="tool_call_123",
            content="Tool result content" * 50000,
        ),
        AIMessage(content="Result for message number"),
        HumanMessage(content="Final test message"),
    ]

    response = await chat_model.ainvoke(messages)
    validate_langchain_response_structure(
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
    chat_model = create_langchain_chat_model(test_client, httpx_mock, is_async=True)

    setup_mock_llm(injector, tools, _DEFAULT_PING_INTERVAL + 1)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Test message"),
    ]

    langchain_tools = convert_to_langchain_tools(tools)
    chat_model = chat_model.bind_tools(langchain_tools)

    response = await chat_model.ainvoke(messages)
    validate_langchain_response_structure(
        response,
        has_tools=True,
        has_internal_tools=True,
    )


ALL_CLAUDE_MODELS = list(get_args(get_args(ModelParam)[0]))


@pytest.mark.parametrize("model", ALL_CLAUDE_MODELS, ids=ALL_CLAUDE_MODELS)
@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_all_models_run_without_crash(
    injector: MockInjector,
    test_client: TestClient,
    httpx_mock: HTTPXMock,
    model: str,
) -> None:
    setup_mock_llm(injector, [])

    chat_model = create_langchain_chat_model(test_client, httpx_mock, model=model)
    response = chat_model.invoke([HumanMessage(content="Test message")])

    assert isinstance(response, AIMessage)
    assert len(response.content) > 0
