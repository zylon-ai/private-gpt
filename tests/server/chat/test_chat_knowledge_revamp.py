import uuid
from typing import Any
from unittest.mock import Mock

import pytest
from httpx import AsyncClient
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import ToolSelection

from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.chat.input_models import (
    Citations,
    MessageInput,
    PromptConfig,
    System,
    SystemExtensions,
    ToolSpecBody,
)
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.events.models import Message, ToolUseBlock
from private_gpt.server.chat.chat_router import ChatBody
from private_gpt.server.utils.artifact_input import IngestedArtifact
from tests.fixtures.mock_function_llm import get_mock_function_calling_llm
from tests.fixtures.mock_injector import MockInjector


class PromptCapture:
    """Helper class to capture system prompts sent to the LLM."""

    def __init__(self) -> None:
        self.captured_messages: list[ChatMessage] = []
        self.system_prompt: str = ""


async def mock_llm_with_capture(
    injector: MockInjector,
    prompt_capture: PromptCapture,
    deltas: list[list[str | ToolSelection]] | None = None,
) -> None:
    """Configure mock LLM and capture the messages sent to it."""
    deltas = deltas or [["Default response"]]
    mock_llm_instance = get_mock_function_calling_llm(deltas)

    # Wrap astream_chat_with_tools to capture messages
    original_astream = mock_llm_instance.astream_chat_with_tools

    async def capturing_astream(
        tools: Any,
        user_msg: Any = None,
        chat_history: list[ChatMessage] | None = None,
        **kwargs: Any,
    ) -> Any:
        # Capture the chat history (which includes system messages)
        if chat_history:
            prompt_capture.captured_messages.extend(chat_history)
            # Extract system prompt from messages
            system_messages = [
                msg for msg in chat_history if msg.role.value == "system"
            ]
            if system_messages:
                prompt_capture.system_prompt = "\n".join(
                    [msg.content or "" for msg in system_messages]
                )

        # Call original - await the coroutine to get the async generator
        gen = await original_astream(tools, user_msg, chat_history, **kwargs)
        async for response in gen:
            yield response

    async def coro(*args, **kwargs):
        return capturing_astream(*args, **kwargs)

    mock_llm_instance.astream_chat_with_tools = coro

    llm_component = injector.get(LLMComponent)
    llm_component.get_llm = Mock(return_value=mock_llm_instance)
    injector.bind_mock(LLMComponent, llm_component)


def create_tool_definition(
    name: str,
    tool_type: str | None = None,
    description: str | None = None,
    input_schema: dict[str, Any] | None = None,
) -> ToolSpecBody:
    """Create a tool definition with proper structure."""
    return ToolSpecBody(
        name=name,
        type=tool_type,
        description=description,
        input_schema=input_schema,
    )


@pytest.mark.anyio
async def test_no_tools_no_system_prompt(
    async_test_client: AsyncClient,
    injector: MockInjector,
) -> None:
    """Test that when no tools are provided."""
    prompt_capture = PromptCapture()
    await mock_llm_with_capture(injector, prompt_capture)

    body = ChatBody(
        messages=[MessageInput(content="Hello", role="user")],
        system=System(),
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())
    assert response.status_code == 200

    system_prompt_lower = prompt_capture.system_prompt.lower()
    has_tool_content = any(
        keyword in system_prompt_lower
        for keyword in [
            "list_files",
            "get_content",
            "knowledge_search",
            "semantic_search",
        ]
    )
    assert not has_tool_content


@pytest.mark.anyio
async def test_default_prompt_flag_is_noop_in_context_stack(
    async_test_client: AsyncClient,
    injector: MockInjector,
) -> None:
    """use_default_prompt is deprecated; no legacy templates injected."""
    prompt_capture = PromptCapture()
    await mock_llm_with_capture(injector, prompt_capture)

    body = ChatBody(
        messages=[MessageInput(content="Hello", role="user")],
        system=System(text="Be concise.", use_default_prompt=True),
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())
    assert response.status_code == 200

    system_prompt_lower = prompt_capture.system_prompt.lower()
    assert "be concise." in system_prompt_lower
    assert "current date:" in system_prompt_lower
    assert "<tool_restrictions>" not in system_prompt_lower


@pytest.mark.anyio
async def test_online_search_tool_system_prompt_injection(
    async_test_client: AsyncClient,
    injector: MockInjector,
) -> None:
    """Test online_search injects instructions when tool_instructions=True."""
    prompt_capture = PromptCapture()
    await mock_llm_with_capture(injector, prompt_capture)

    body = ChatBody(
        messages=[MessageInput(content="Search the web", role="user")],
        system=System(prompt=PromptConfig(tools=True)),
        tools=[
            create_tool_definition(
                name="online_search",
                tool_type="web_search_v1",
            )
        ],
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())
    assert response.status_code == 200

    system_prompt_lower = prompt_capture.system_prompt.lower()
    # web_search template should be injected when tool_instructions=True
    assert "online_search" in system_prompt_lower or "web" in system_prompt_lower


@pytest.mark.anyio
async def test_prompt_features_disabled_no_guidelines(
    async_test_client: AsyncClient,
    injector: MockInjector,
) -> None:
    """PromptConfig defaults to all False — no platform prompts injected by default."""
    prompt_capture = PromptCapture()
    await mock_llm_with_capture(injector, prompt_capture)

    body = ChatBody(
        messages=[MessageInput(content="Hello", role="user")],
        # prompt features default to all False
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())
    assert response.status_code == 200

    system_prompt_lower = prompt_capture.system_prompt.lower()
    # No platform prompts should appear
    assert "response_formatting" not in system_prompt_lower
    assert "knowledge_base_operations" not in system_prompt_lower


@pytest.mark.anyio
async def test_citations_enabled_system_prompt_injection_without_sources(
    async_test_client: AsyncClient,
    injector: MockInjector,
) -> None:
    """Test that citations enabled adds citation prompt."""
    collection = str(uuid.uuid4())
    prompt_capture = PromptCapture()

    await mock_llm_with_capture(injector, prompt_capture)

    body = ChatBody(
        messages=[MessageInput(content="Search for information", role="user")],
        system=System(
            citations=Citations(enabled=True),
            extensions={SystemExtensions.ZYLON},
        ),
        tools=[
            create_tool_definition(
                name="knowledge_search",
                tool_type="semantic_search_v1",
            )
        ],
        tool_context=[
            IngestedArtifact(context_filter=ContextFilter(collection=collection))
        ],
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    assert response.status_code == 200

    system_prompt_lower = prompt_capture.system_prompt.lower()
    assert "citation" not in system_prompt_lower


@pytest.mark.anyio
async def test_tool_autonomous_selection_without_tool_choice(
    async_test_client: AsyncClient,
    injector: MockInjector,
) -> None:
    """Test that agent can autonomously select tools without explicit tool_choice."""
    collection = str(uuid.uuid4())
    artifact = str(uuid.uuid4())

    # Ingest test data
    ingest_response = await async_test_client.post(
        "/v1/artifacts/ingest",
        json={
            "metadata": {},
            "input": {
                "type": "text",
                "value": "Test document content",
            },
            "collection": collection,
            "artifact": artifact,
        },
    )
    assert ingest_response.status_code == 200

    # Mock LLM to return tool selection
    prompt_capture = PromptCapture()
    tool_deltas: list[list[str | ToolSelection]] = [
        [
            ToolSelection(
                tool_id="list_files",
                tool_name="list_files",
                tool_kwargs={"query": "documents"},
            )
        ],
        ["Found documents"],
    ]
    await mock_llm_with_capture(injector, prompt_capture, deltas=tool_deltas)

    body = ChatBody(
        messages=[MessageInput(content="What files do I have?", role="user")],
        tools=[
            create_tool_definition(
                name="list_files",
                description="List files",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            )
        ],
        tool_context=[
            IngestedArtifact(context_filter=ContextFilter(collection=collection))
        ],
        # No tool_choice specified - agent should autonomously select
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    assert response.status_code == 200
    completion: Message = Message.model_validate(response.json())
    assert any(isinstance(block, ToolUseBlock) for block in completion.content)

    # Cleanup
    await async_test_client.post(
        "/v1/artifacts/delete",
        json={"collection": collection, "artifact": artifact},
    )


@pytest.mark.anyio
async def test_knowledge_search_tool_instructions_injected(
    async_test_client: AsyncClient,
    injector: MockInjector,
) -> None:
    """With tools=True, tool instructions are injected under the caller-given name."""
    collection = str(uuid.uuid4())
    prompt_capture = PromptCapture()

    await mock_llm_with_capture(injector, prompt_capture)

    body = ChatBody(
        messages=[MessageInput(content="Tell me about the documents", role="user")],
        system=System(prompt=PromptConfig(tools=True)),
        tools=[
            create_tool_definition(
                name="knowledge_search",
                tool_type="semantic_search_v1",
            )
        ],
        tool_context=[
            IngestedArtifact(context_filter=ContextFilter(collection=collection))
        ],
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())

    assert response.status_code == 200
    system_prompt_lower = prompt_capture.system_prompt.lower()
    # Instructions are injected under the caller-assigned name, not the internal type.
    assert "knowledge_search" in system_prompt_lower
    assert "semantic_search_v1" not in system_prompt_lower


@pytest.mark.anyio
async def test_external_tool_without_tool_context_succeeds(
    async_test_client: AsyncClient,
    injector: MockInjector,
) -> None:
    """Test that external (non-internal) tools work without tool_context."""
    prompt_capture = PromptCapture()
    tool_deltas: list[list[str | ToolSelection]] = [
        [
            ToolSelection(
                tool_id="custom_tool",
                tool_name="custom_tool",
                tool_kwargs={"param": "value"},
            )
        ],
        ["Tool executed successfully"],
    ]
    await mock_llm_with_capture(injector, prompt_capture, deltas=tool_deltas)

    body = ChatBody(
        messages=[MessageInput(content="Use custom tool", role="user")],
        tools=[
            create_tool_definition(
                name="custom_tool",
                description="A custom external tool",
                input_schema={
                    "type": "object",
                    "properties": {"param": {"type": "string"}},
                },
            )
        ],
        # No tool_context - external tool doesn't need it
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())
    assert response.status_code == 200
    completion: Message = Message.model_validate(response.json())
    assert any(isinstance(block, ToolUseBlock) for block in completion.content)


@pytest.mark.anyio
async def test_tool_instructions_external_tool_with_explicit_instructions(
    async_test_client: AsyncClient,
    injector: MockInjector,
) -> None:
    """External tool with explicit instructions field gets them injected."""
    prompt_capture = PromptCapture()
    await mock_llm_with_capture(injector, prompt_capture)

    tool_with_instructions = ToolSpecBody(
        name="my_custom_tool",
        description="A tool with custom instructions",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        instructions="Always use this tool when the user asks about custom topics.",
    )

    body = ChatBody(
        messages=[MessageInput(content="Use my tool", role="user")],
        system=System(prompt=PromptConfig(tools=True)),
        tools=[tool_with_instructions],
    )
    response = await async_test_client.post("/v1/messages", json=body.model_dump())
    assert response.status_code == 200

    system_prompt_lower = prompt_capture.system_prompt.lower()
    assert "always use this tool" in system_prompt_lower
