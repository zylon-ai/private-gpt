import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult, TextContent

from private_gpt.components.engines.chat.utils.tool_utils import execute_tool_call
from private_gpt.components.tools.remote_execution import (
    ToolExecutionResponse,
    rebuild_tool_from_spec,
)
from private_gpt.events.models import McpTokensRefreshedEvent
from private_gpt.server.mcp.config import McpServerConfig
from private_gpt.server.mcp.mcp_service import (
    McpToolDefinition,
    rebuild_mcp_tool,
)


def _config() -> McpServerConfig:
    return McpServerConfig(
        name="tools",
        url="https://mcp.example.com",
        artifact_id="artifact-123",
        authorization_token="access-before-sentinel",
        refresh_token="refresh-before-sentinel",
        client_id="client-id",
        token_endpoint="https://auth.example.com/token",
    )


def _event() -> McpTokensRefreshedEvent:
    return McpTokensRefreshedEvent(
        artifact_id="artifact-123",
        previous_refresh_token="refresh-before-sentinel",
        authorization_token="access-after-sentinel",
        refresh_token="refresh-after-sentinel",
    )


def _result() -> CallToolResult:
    return CallToolResult(
        content=[TextContent(text="normal MCP content")],
    )


def test_tool_execution_response_roundtrips_refresh_event() -> None:
    response = ToolExecutionResponse(
        tool_name="lookup",
        tool_id="tool-1",
        result_content=[{"type": "text", "text": "normal MCP content"}],
        tool_message={
            "role": "tool",
            "content": "normal MCP content",
            "additional_kwargs": {"tool_call_id": "tool-1"},
        },
        internal_events=[_event()],
    )

    restored = ToolExecutionResponse.model_validate_json(response.model_dump_json())

    assert restored == response
    assert restored.internal_events == [_event()]


async def _execute_rebuilt_tool(
    call_tool: AsyncMock,
    emit_refresh: bool = True,
) -> tuple[object, object, list[McpTokensRefreshedEvent]]:
    client = MagicMock()
    client.list_tools = AsyncMock(
        return_value=[
            McpToolDefinition(
                name="lookup",
                description="Look something up",
                input_schema={"type": "object", "properties": {}},
            )
        ]
    )
    client.call_tool = call_tool
    client.close = AsyncMock()
    callback_holder: dict[str, object] = {}

    def make_client(_config: McpServerConfig, *, emit_event: object) -> MagicMock:
        callback_holder["emit_event"] = emit_event
        return client

    async def emit_refresh_then_call(*args: object, **kwargs: object) -> object:
        if emit_refresh:
            callback = callback_holder["emit_event"]
            assert callable(callback)
            callback(_event())
        return await call_tool(*args, **kwargs)

    with patch(
        "private_gpt.server.mcp.mcp_service.McpClient",
        side_effect=make_client,
    ):
        spec = rebuild_mcp_tool(
            config=_config(),
            tool_name="lookup",
            name="lookup",
            type=None,
            description="Look something up",
            input_schema={"type": "object", "properties": {}},
        )
        rebuilt = await rebuild_tool_from_spec(spec)
        client.call_tool = emit_refresh_then_call
        result, message, events = await execute_tool_call(
            tool=rebuilt,
            tool_name="lookup",
            tool_id="tool-1",
            tool_kwargs={},
            state_ctx=None,
        )
        return result, message, events


@pytest.mark.asyncio
async def test_mcp_tool_result_unwraps_and_returns_refresh_event_separately() -> None:
    result, message, events = await _execute_rebuilt_tool(
        AsyncMock(return_value=_result())
    )

    assert result.tool_output.raw_output == _result()
    assert result.tool_output.content == "normal MCP content"
    assert result.tool_output.is_error is False
    assert events == [_event()]
    serialized_message = json.dumps(message.model_dump(mode="json"), default=str)
    for sentinel in (
        "refresh-before-sentinel",
        "access-after-sentinel",
        "refresh-after-sentinel",
    ):
        assert sentinel not in serialized_message


@pytest.mark.asyncio
async def test_mcp_tool_failure_after_refresh_returns_event_and_error() -> None:
    result, message, events = await _execute_rebuilt_tool(
        AsyncMock(side_effect=RuntimeError("tool failed"))
    )

    assert result.tool_output.is_error is True
    assert result.tool_output.content == "tool failed"
    assert events == [_event()]
    serialized_message = json.dumps(message.model_dump(mode="json"), default=str)
    assert "tool failed" in serialized_message
    assert "refresh-after-sentinel" not in serialized_message


@pytest.mark.asyncio
async def test_mcp_tool_failure_before_refresh_keeps_existing_exception_path() -> None:
    result, message, events = await _execute_rebuilt_tool(
        AsyncMock(side_effect=RuntimeError("tool failed before refresh")),
        emit_refresh=False,
    )

    assert result.tool_output.is_error is True
    assert result.tool_output.content == "tool failed before refresh"
    assert events == []
    assert "refresh-after-sentinel" not in json.dumps(
        message.model_dump(mode="json"), default=str
    )
