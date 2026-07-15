from unittest.mock import AsyncMock, patch

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
    assert restored.execution_metadata.rebuild_kwargs["config"] == config
    assert restored.execution_metadata.rebuild_kwargs["tool_name"] == "lookup"


@pytest.mark.asyncio
async def test_rebuild_mcp_tool_reconnects_and_selects_named_tool() -> None:
    config = McpServerConfig(name="tools", url="https://mcp.example.com")
    spec = mcp_tool_to_spec(config, _function_tool())

    with patch(
        "private_gpt.server.mcp.mcp_service.McpClient.list_tools",
        new=AsyncMock(return_value=[_function_tool()]),
    ):
        rebuilt = await rebuild_tool_from_spec(spec)

    assert rebuilt.metadata.name == "lookup"
