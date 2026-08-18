from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult, TextContent

from private_gpt.components.tools.remote_execution import (
    ToolExecutionRequest,
    ToolExecutionResponse,
    ToolExecutor,
)
from private_gpt.events.models import McpTokensRefreshedEvent
from private_gpt.server.mcp.config import McpServerConfig
from private_gpt.server.mcp.mcp_service import McpToolDefinition, mcp_tool_to_spec


@pytest.mark.asyncio
async def test_tool_execution_carries_refresh_event_outside_model_content() -> None:
    config = McpServerConfig(name="tools", url="https://mcp.example.com")
    tool = McpToolDefinition(
        name="lookup",
        description="Look something up",
        input_schema={"type": "object", "properties": {}},
    )
    event = McpTokensRefreshedEvent(
        name="tools",
        url=config.url,
        previous_refresh_token="refresh-before-sentinel",
        authorization_token="access-after-sentinel",
        refresh_token="refresh-after-sentinel",
        metadata={"artifact_id": "artifact-123"},
    )
    client = MagicMock()
    client.list_tools = AsyncMock(return_value=[tool])
    client.call_tool = AsyncMock(
        return_value=CallToolResult(content=[TextContent(text="normal MCP content")])
    )
    client.token_refresh_event.return_value = event
    client.close = AsyncMock()

    with patch("private_gpt.server.mcp.mcp_service.McpClient", return_value=client):
        response = await ToolExecutor().execute(
            ToolExecutionRequest(
                tool_id="tool-1",
                tool_name="lookup",
                tool_spec=mcp_tool_to_spec(config, tool),
            )
        )

    restored = ToolExecutionResponse.model_validate_json(response.model_dump_json())
    assert restored.internal_events == [event]
    serialized_message = response.tool_message.model_dump_json()
    assert "normal MCP content" in serialized_message
    for secret in (
        "refresh-before-sentinel",
        "access-after-sentinel",
        "refresh-after-sentinel",
    ):
        assert secret not in serialized_message
    client.close.assert_awaited_once()
