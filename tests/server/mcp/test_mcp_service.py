from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.tools import FunctionTool

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.tools.remote_execution import rebuild_tool_from_spec
from private_gpt.server.mcp.config import McpServerConfig
from private_gpt.server.mcp.mcp_service import mcp_tool_to_spec


async def _remote_tool(query: str) -> str:
    return f"result:{query}"


def _function_tool() -> FunctionTool:
    return FunctionTool.from_defaults(
        name="lookup",
        description="Look something up",
        async_fn=_remote_tool,
    )


def test_mcp_tool_spec_json_roundtrip_preserves_rebuild_config() -> None:
    config = McpServerConfig(name="tools", url="https://mcp.example.com")
    spec = mcp_tool_to_spec(config, _function_tool())

    restored = ToolSpec.model_validate_json(spec.model_dump_json())

    assert restored.execution_metadata is not None
    kwargs = restored.execution_metadata.rebuild_kwargs
    assert kwargs["config"] == config
    assert kwargs["tool_name"] == "lookup"
    assert kwargs["name"] == "lookup"
    assert kwargs["description"] == "Look something up"
    assert kwargs["input_schema"] == spec.input_schema


@pytest.mark.asyncio
async def test_rebuilt_mcp_tool_uses_task_scoped_client() -> None:
    config = McpServerConfig(name="tools", url="https://mcp.example.com")
    spec = mcp_tool_to_spec(config, _function_tool())
    client = MagicMock()
    client.list_tools = AsyncMock(return_value=[_function_tool()])
    client.close = AsyncMock()

    with patch("private_gpt.server.mcp.mcp_service.McpClient", return_value=client):
        rebuilt = await rebuild_tool_from_spec(spec)
        result = await rebuilt.acall(query="platform")

    assert result.raw_output == "result:platform"
    client.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_rebuilt_mcp_tool_closes_client_when_tool_is_missing() -> None:
    config = McpServerConfig(name="tools", url="https://mcp.example.com")
    spec = mcp_tool_to_spec(config, _function_tool())
    client = MagicMock()
    client.list_tools = AsyncMock(return_value=[])
    client.close = AsyncMock()

    with patch("private_gpt.server.mcp.mcp_service.McpClient", return_value=client):
        rebuilt = await rebuild_tool_from_spec(spec)
        with pytest.raises(ValueError, match="no longer available"):
            await rebuilt.acall(query="platform")

    client.close.assert_awaited_once()
