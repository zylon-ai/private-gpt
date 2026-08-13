from unittest.mock import AsyncMock, MagicMock, patch

import httpx2
import pytest
from mcp.types import ListToolsResult, Tool

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.tools.remote_execution import rebuild_tool_from_spec
from private_gpt.events.models import McpTokensRefreshedEvent
from private_gpt.server.mcp._runtime import PersistentMCPClient, RefreshTokenAuth
from private_gpt.server.mcp.config import McpServerConfig, McpServerToolConfig
from private_gpt.server.mcp.mcp_service import (
    McpClient,
    McpToolDefinition,
    McpToolExecutionResult,
    mcp_tool_to_spec,
)


def _remote_tool_def() -> McpToolDefinition:
    return McpToolDefinition(
        name="lookup",
        description="Look something up",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )


def test_mcp_refresh_token_requires_client_and_endpoint() -> None:
    with pytest.raises(ValueError, match="client_id, token_endpoint"):
        McpServerConfig(
            url="https://mcp.example.com",
            refresh_token="refresh-token",
        )


@pytest.mark.asyncio
async def test_refresh_token_auth_refreshes_and_retries_after_401() -> None:
    refreshed: list[tuple[str, str]] = []
    auth = RefreshTokenAuth(
        access_token="expired-access-token",
        refresh_token="old-refresh-token",
        client_id="client-id",
        token_endpoint="https://auth.example.com/token",
        resource="https://mcp.example.com",
        on_tokens_refreshed=lambda access, refresh: refreshed.append((access, refresh)),
    )
    flow = auth.async_auth_flow(httpx2.Request("POST", "https://mcp.example.com/mcp"))

    request = await anext(flow)
    assert request.headers["Authorization"] == "Bearer expired-access-token"

    refresh_request = await flow.asend(httpx2.Response(401, request=request))
    assert str(refresh_request.url) == "https://auth.example.com/token"
    assert refresh_request.content.decode() == (
        "grant_type=refresh_token&refresh_token=old-refresh-token&"
        "client_id=client-id&resource=https%3A%2F%2Fmcp.example.com"
    )

    retry_request = await flow.asend(
        httpx2.Response(
            200,
            json={
                "access_token": "new-access-token",
                "refresh_token": "new-refresh-token",
                "token_type": "Bearer",
            },
            request=refresh_request,
        )
    )
    assert retry_request.headers["Authorization"] == "Bearer new-access-token"
    assert refreshed == [("new-access-token", "new-refresh-token")]


def test_mcp_tool_spec_json_roundtrip_preserves_rebuild_config() -> None:
    config = McpServerConfig(
        name="tools",
        url="https://mcp.example.com",
        artifact_id="artifact-123",
    )
    restored_config = McpServerConfig.model_validate_json(config.model_dump_json())
    assert restored_config.artifact_id == "artifact-123"

    spec = mcp_tool_to_spec(config, _remote_tool_def())

    restored = ToolSpec.model_validate_json(spec.model_dump_json())

    assert restored.execution_metadata is not None
    kwargs = restored.execution_metadata.rebuild_kwargs
    assert kwargs["config"] == config
    assert kwargs["config"].artifact_id == "artifact-123"
    assert kwargs["tool_name"] == "lookup"
    assert kwargs["name"] == "lookup"
    assert kwargs["description"] == "Look something up"
    assert kwargs["input_schema"] == spec.input_schema


def test_mcp_client_emits_refresh_event_and_updates_memory() -> None:
    config = McpServerConfig(
        name="tools",
        url="https://mcp.example.com",
        artifact_id="artifact-123",
        authorization_token="access-before",
        refresh_token="refresh-before",
        client_id="client-id",
        token_endpoint="https://auth.example.com/token",
    )
    emitted: list[McpTokensRefreshedEvent] = []
    runtime_client = MagicMock()
    runtime_factory = MagicMock(return_value=runtime_client)

    with patch(
        "private_gpt.server.mcp.mcp_service._load_runtime",
        return_value=runtime_factory,
    ):
        McpClient(config, emit_event=emitted.append)

    callback = runtime_factory.call_args.kwargs["on_tokens_refreshed"]
    callback("access-after", "refresh-after")

    assert config.authorization_token == "access-after"
    assert config.refresh_token == "refresh-after"
    assert emitted == [
        McpTokensRefreshedEvent(
            artifact_id="artifact-123",
            previous_refresh_token="refresh-before",
            authorization_token="access-after",
            refresh_token="refresh-after",
        )
    ]


def test_mcp_client_without_artifact_id_only_updates_memory() -> None:
    config = McpServerConfig(
        url="https://mcp.example.com",
        authorization_token="access-before",
        refresh_token="refresh-before",
        client_id="client-id",
        token_endpoint="https://auth.example.com/token",
    )
    emitted: list[McpTokensRefreshedEvent] = []
    runtime_factory = MagicMock(return_value=MagicMock())

    with patch(
        "private_gpt.server.mcp.mcp_service._load_runtime",
        return_value=runtime_factory,
    ):
        McpClient(config, emit_event=emitted.append)

    runtime_factory.call_args.kwargs["on_tokens_refreshed"](
        "access-after", "refresh-after"
    )

    assert config.authorization_token == "access-after"
    assert config.refresh_token == "refresh-after"
    assert emitted == []


@pytest.mark.asyncio
async def test_refresh_without_replacement_token_keeps_existing_refresh_token() -> None:
    refreshed: list[tuple[str, str]] = []
    auth = RefreshTokenAuth(
        access_token="expired-access-token",
        refresh_token="old-refresh-token",
        client_id="client-id",
        token_endpoint="https://auth.example.com/token",
        on_tokens_refreshed=lambda access, refresh: refreshed.append((access, refresh)),
    )
    flow = auth.async_auth_flow(httpx2.Request("POST", "https://mcp.example.com/mcp"))
    request = await anext(flow)
    refresh_request = await flow.asend(httpx2.Response(401, request=request))

    retry_request = await flow.asend(
        httpx2.Response(
            200,
            json={"access_token": "new-access-token", "token_type": "Bearer"},
            request=refresh_request,
        )
    )

    assert retry_request.headers["Authorization"] == "Bearer new-access-token"
    assert refreshed == [("new-access-token", "old-refresh-token")]


@pytest.mark.asyncio
async def test_401_refresh_emits_one_mcp_event() -> None:
    config = McpServerConfig(
        url="https://mcp.example.com/mcp",
        artifact_id="artifact-123",
        authorization_token="expired-access-token",
        refresh_token="old-refresh-token",
        client_id="client-id",
        token_endpoint="https://auth.example.com/token",
    )
    emitted: list[McpTokensRefreshedEvent] = []

    with patch(
        "private_gpt.server.mcp.mcp_service._load_runtime",
        return_value=PersistentMCPClient,
    ):
        client = McpClient(config, emit_event=emitted.append)

    assert client.client is not None
    assert client.client.auth is not None
    flow = client.client.auth.async_auth_flow(
        httpx2.Request("POST", "https://mcp.example.com/mcp")
    )
    request = await anext(flow)
    refresh_request = await flow.asend(httpx2.Response(401, request=request))
    await flow.asend(
        httpx2.Response(
            200,
            json={
                "access_token": "new-access-token",
                "refresh_token": "new-refresh-token",
                "token_type": "Bearer",
            },
            request=refresh_request,
        )
    )

    assert emitted == [
        McpTokensRefreshedEvent(
            artifact_id="artifact-123",
            previous_refresh_token="old-refresh-token",
            authorization_token="new-access-token",
            refresh_token="new-refresh-token",
        )
    ]


@pytest.mark.asyncio
async def test_rebuilt_mcp_tool_uses_task_scoped_client() -> None:
    config = McpServerConfig(name="tools", url="https://mcp.example.com")
    spec = mcp_tool_to_spec(config, _remote_tool_def())
    client = MagicMock()
    client.list_tools = AsyncMock(return_value=[_remote_tool_def()])
    client.call_tool = AsyncMock(return_value="result:platform")
    client.close = AsyncMock()

    with patch("private_gpt.server.mcp.mcp_service.McpClient", return_value=client):
        rebuilt = await rebuild_tool_from_spec(spec)
        result = await rebuilt.acall(query="platform")

    assert isinstance(result.raw_output, McpToolExecutionResult)
    assert result.raw_output.result == "result:platform"
    client.call_tool.assert_awaited_once_with("lookup", {"query": "platform"})
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_rebuilt_mcp_tool_closes_client_when_tool_is_missing() -> None:
    config = McpServerConfig(name="tools", url="https://mcp.example.com")
    spec = mcp_tool_to_spec(config, _remote_tool_def())
    client = MagicMock()
    client.list_tools = AsyncMock(return_value=[])
    client.close = AsyncMock()

    with patch("private_gpt.server.mcp.mcp_service.McpClient", return_value=client):
        rebuilt = await rebuild_tool_from_spec(spec)
        with pytest.raises(ValueError, match="no longer available"):
            await rebuilt.acall(query="platform")

    client.close.assert_awaited_once()


def test_mcp_tool_to_spec_preserves_untyped_original_schema() -> None:
    """n8n-style untyped properties must stay untyped (not rewritten to string)."""
    original_schema = {
        "type": "object",
        "properties": {
            "object_param": {
                "description": (
                    "send whatever input the user gave you in the form of a json array"
                )
            }
        },
        "required": ["object_param"],
        "additionalProperties": True,
        "$schema": "http://json-schema.org/draft-07/schema#",
    }
    tool = McpToolDefinition(
        name="Call_test_tool_object_",
        description="Call this tool when prompted to do so",
        input_schema=original_schema,
    )
    config = McpServerConfig(name="Test MCP", url="https://n8n.example.com/mcp/test")
    spec = mcp_tool_to_spec(config, tool)

    assert spec.input_schema is not None
    prop = spec.input_schema["properties"]["object_param"]
    assert "type" not in prop, f"original untyped schema was rewritten: {prop}"
    assert (
        prop["description"]
        == original_schema["properties"]["object_param"]["description"]
    )


def test_mcp_tool_definition_from_typed_mcp_tool() -> None:
    original_schema = {
        "type": "object",
        "properties": {
            "object_param": {
                "description": (
                    "send whatever input the user gave you in the form of a json array"
                )
            }
        },
        "required": ["object_param"],
        "additionalProperties": True,
    }
    remote_tool = Tool(
        name="Call_test_tool_object_",
        description="Call this tool when prompted to do so",
        inputSchema=original_schema,
    )
    definition = McpToolDefinition.from_mcp_tool(remote_tool)
    assert definition.name == "Call_test_tool_object_"
    assert definition.description == "Call this tool when prompted to do so"
    assert definition.input_schema == original_schema
    assert "type" not in definition.input_schema["properties"]["object_param"]


@pytest.mark.asyncio
async def test_mcp_client_list_tools_preserves_remote_input_schema() -> None:
    config = McpServerConfig(
        name="Test MCP",
        url="https://n8n.example.com/mcp/test",
        tool_configuration=McpServerToolConfig(
            enabled=True,
            allowed_tools=["Call_test_tool_object_"],
        ),
    )
    original_schema = {
        "type": "object",
        "properties": {
            "object_param": {
                "description": (
                    "send whatever input the user gave you in the form of a json array"
                )
            }
        },
        "required": ["object_param"],
        "additionalProperties": True,
        "$schema": "http://json-schema.org/draft-07/schema#",
    }
    remote_tool = Tool(
        name="Call_test_tool_object_",
        description="Call this tool when prompted to do so",
        inputSchema=original_schema,
    )
    ignored_tool = Tool(
        name="other_tool",
        description="ignored",
        inputSchema={"type": "object", "properties": {}},
    )
    runtime_client = MagicMock()
    runtime_client.list_tools = AsyncMock(
        return_value=ListToolsResult(tools=[remote_tool, ignored_tool])
    )

    with patch(
        "private_gpt.server.mcp.mcp_service._load_runtime",
        return_value=MagicMock(return_value=runtime_client),
    ):
        client = McpClient(config)
        tools = await client.list_tools()

    assert len(tools) == 1
    assert tools[0].name == "Call_test_tool_object_"
    assert tools[0].input_schema == original_schema
    assert "type" not in tools[0].input_schema["properties"]["object_param"]
