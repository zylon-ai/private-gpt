from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.components.chat.models.chat_config_models import (
    ChatRequest,
    ToolSpec,
)
from private_gpt.components.context.models.context_layer import ToolDefinitionsLayer
from private_gpt.components.context.models.context_stack import ContextStack
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.engines.chat.models.chat_state import ChatInputState
from private_gpt.components.tools.remote_execution import apply_tool_spec_update
from private_gpt.events.event_errors import Errors
from private_gpt.events.models import (
    McpTokensRefreshedEvent,
    McpTokensRefreshFailedEvent,
    RawContentBlockStartEvent,
)
from private_gpt.server.chat.interceptors.mcp_interceptor import (
    McpRequestInterceptor,
)
from private_gpt.server.mcp.config import McpServerConfig
from private_gpt.server.mcp.mcp_service import (
    MCP_PREVIOUS_REFRESH_TOKEN_KEY,
    MCP_REFRESH_FAILED_KEY,
    McpToolDefinition,
    mcp_tool_to_spec,
)


def _discovery_setup() -> tuple[
    McpServerConfig,
    MagicMock,
    MagicMock,
    McpRequestInterceptor,
    MagicMock,
]:
    config = McpServerConfig(
        url="https://mcp.example.com",
        refresh_token="refresh-before",
        client_id="client-id",
        metadata={"artifact_id": "artifact-123"},
    )
    client = MagicMock()
    client.close = AsyncMock()
    mcp_service = MagicMock()
    mcp_service.create_client.return_value = client
    interceptor = McpRequestInterceptor(mcp_service)
    request = ChatRequest(messages=[], mcp_servers=[config])
    context = MagicMock()
    context.phase = InterceptorPhase.VALIDATION
    context.state = SimpleNamespace(
        input=SimpleNamespace(request=request, context_stack=MagicMock()),
        original_input=None,
    )
    context.emit_event = MagicMock()
    return config, client, mcp_service, interceptor, context


def _record_refresh(config: McpServerConfig) -> None:
    config.authorization_token = "access-after"
    config.refresh_token = "refresh-after"
    config.metadata[MCP_PREVIOUS_REFRESH_TOKEN_KEY] = "refresh-before"


@pytest.mark.asyncio
async def test_discovery_emits_refreshed_tokens_as_a_chat_event() -> None:
    config, client, mcp_service, interceptor, context = _discovery_setup()

    async def list_tools() -> list[McpToolDefinition]:
        _record_refresh(config)
        return [
            McpToolDefinition(
                name="lookup",
                description="Look something up",
                input_schema={"type": "object", "properties": {}},
            )
        ]

    client.list_tools = AsyncMock(side_effect=list_tools)
    event = McpTokensRefreshedEvent(
        name="mcp",
        url="https://mcp.example.com",
        previous_refresh_token="refresh-before",
        authorization_token="access-after",
        refresh_token="refresh-after",
        metadata={"artifact_id": "artifact-123"},
    )

    await interceptor.intercept(context)

    mcp_service.create_client.assert_called_once_with(config)
    context.emit_event.assert_called_once_with(event)
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_discovery_emits_refreshed_tokens_before_wrapping_error() -> None:
    config, client, _, interceptor, context = _discovery_setup()
    error = RuntimeError("discovery failed after refresh")

    async def list_tools() -> None:
        _record_refresh(config)
        raise error

    client.list_tools = AsyncMock(side_effect=list_tools)
    event = McpTokensRefreshedEvent(
        name="mcp",
        url="https://mcp.example.com",
        previous_refresh_token="refresh-before",
        authorization_token="access-after",
        refresh_token="refresh-after",
        metadata={"artifact_id": "artifact-123"},
    )

    with pytest.raises(Errors.InvalidRequest) as exc_info:
        await interceptor.intercept(context)

    assert exc_info.value.__cause__ is error
    context.emit_event.assert_called_once_with(event)
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_discovery_emits_refresh_failed_before_wrapping_error() -> None:
    config, client, _, interceptor, context = _discovery_setup()
    error = RuntimeError("refresh failed")
    config.metadata[MCP_REFRESH_FAILED_KEY] = True
    client.list_tools = AsyncMock(side_effect=error)
    event = McpTokensRefreshFailedEvent(
        name="mcp",
        url="https://mcp.example.com",
        error="MCP OAuth token refresh failed",
        metadata={"artifact_id": "artifact-123"},
    )

    with pytest.raises(Errors.InvalidRequest) as exc_info:
        await interceptor.intercept(context)

    assert exc_info.value.__cause__ is error
    context.emit_event.assert_called_once_with(event)
    client.close.assert_awaited_once()


def _mcp_config(tool: ToolSpec) -> McpServerConfig:
    assert tool.execution_metadata is not None
    config = tool.execution_metadata.rebuild_kwargs["config"]
    assert isinstance(config, McpServerConfig)
    return config


def _mcp_tool(name: str, config: McpServerConfig) -> ToolSpec:
    return mcp_tool_to_spec(
        config,
        McpToolDefinition(
            name=name,
            description=None,
            input_schema={"type": "object", "properties": {}},
        ),
    )


@pytest.mark.asyncio
async def test_tool_refresh_is_consumed_by_interceptor_and_persisted() -> None:
    config = McpServerConfig(
        name="tools",
        url="https://mcp.example.com",
        authorization_token="access-before",
        refresh_token="refresh-before",
        client_id="client-id",
    )
    tool = _mcp_tool("lookup", config)
    sibling_tool = _mcp_tool("search", config.model_copy(deep=True))
    updated_tool = _mcp_tool(
        "lookup",
        McpServerConfig(
            name="tools",
            url=config.url,
            authorization_token="access-after",
            refresh_token="refresh-after",
            client_id="client-id",
            metadata={
                "artifact_id": "artifact-123",
                MCP_PREVIOUS_REFRESH_TOKEN_KEY: "refresh-before",
            },
        ),
    )
    input_state = ChatInputState(
        request=ChatRequest(messages=[]),
        context_stack=ContextStack(
            layers=[ToolDefinitionsLayer(tools=[tool, sibling_tool], source="mcp")]
        ),
    )
    state = SimpleNamespace(
        input=input_state,
        original_input=input_state.model_copy(deep=True),
    )
    apply_tool_spec_update(state.input, updated_tool)
    apply_tool_spec_update(state.original_input, updated_tool)
    emitted: list[object] = []
    mcp_service = MagicMock()
    interceptor = McpRequestInterceptor(mcp_service)
    context = MagicMock()
    context.state = state
    context.emit_event.side_effect = emitted.append

    event = RawContentBlockStartEvent.from_text()
    assert await interceptor.intercept_event(event, context) is event

    assert emitted == [
        McpTokensRefreshedEvent(
            name="tools",
            url=config.url,
            previous_refresh_token="refresh-before",
            authorization_token="access-after",
            refresh_token="refresh-after",
            metadata={"artifact_id": "artifact-123"},
        )
    ]
    assert state.original_input is not None
    restored = ChatInputState.model_validate_json(
        state.original_input.model_dump_json()
    )
    for persisted_tool in (
        *state.input.context_stack.all_tools(),
        *restored.context_stack.all_tools(),
    ):
        persisted_config = _mcp_config(persisted_tool)
        assert persisted_config.authorization_token == "access-after"
        assert persisted_config.refresh_token == "refresh-after"
        assert MCP_PREVIOUS_REFRESH_TOKEN_KEY not in persisted_config.metadata
