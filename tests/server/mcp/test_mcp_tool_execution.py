from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from private_gpt.components.tools.remote_execution import (
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolExecutor,
)
from private_gpt.server.chat.interceptors.mcp_interceptor import McpRequestInterceptor
from private_gpt.server.mcp.config import McpServerConfig
from private_gpt.server.mcp.mcp_service import (
    MCP_PREVIOUS_REFRESH_TOKEN_KEY,
    MCP_TOKEN_REFRESH_KEY,
    McpToolDefinition,
    mcp_tool_to_spec,
)


@pytest.mark.asyncio
async def test_tool_execution_returns_mutated_mcp_request_state() -> None:
    config = McpServerConfig(name="tools", url="https://mcp.example.com")
    tool = McpToolDefinition(
        name="lookup",
        description="Look something up",
        input_schema={"type": "object", "properties": {}},
    )
    runtime_client = MagicMock()
    runtime_client.list_tools = AsyncMock(
        return_value=ListToolsResult(
            tools=[
                Tool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema=tool.input_schema,
                )
            ]
        )
    )
    runtime_client.refreshed_tokens = None
    runtime_client.refresh_attempted = False
    runtime_client.close = AsyncMock()

    async def call_tool(*_args: object, **_kwargs: object) -> CallToolResult:
        runtime_client.refreshed_tokens = (
            "access-after-sentinel",
            "refresh-after-sentinel",
            "refresh-before-sentinel",
        )
        return CallToolResult(content=[TextContent(text="normal MCP content")])

    runtime_client.call_tool = AsyncMock(side_effect=call_tool)

    config.authorization_token = "access-before-sentinel"
    config.refresh_token = "refresh-before-sentinel"
    config.client_id = "client-id"
    with patch(
        "private_gpt.server.mcp.mcp_service._load_runtime",
        return_value=MagicMock(return_value=runtime_client),
    ):
        response = await ToolExecutor(
            interceptors=[McpRequestInterceptor(MagicMock())]
        ).execute(
            ToolExecutionRequest(
                tool_id="tool-1",
                tool_name="lookup",
                tool_spec=mcp_tool_to_spec(config, tool),
            )
        )

    restored = ToolExecutionResponse.model_validate_json(response.model_dump_json())
    payload = restored.tool_message.additional_kwargs[MCP_TOKEN_REFRESH_KEY]
    assert payload["previous_refresh_token"] == "refresh-before-sentinel"
    assert payload["authorization_token"] == "access-after-sentinel"
    assert payload["refresh_token"] == "refresh-after-sentinel"
    assert MCP_PREVIOUS_REFRESH_TOKEN_KEY not in config.metadata
    assert restored.tool_message.content == "normal MCP content"
    runtime_client.close.assert_awaited_once()
