from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import ListToolsResult, Tool

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.tools.remote_execution import rebuild_tool_from_spec
from private_gpt.server.mcp.config import McpServerConfig, McpServerToolConfig
from private_gpt.server.mcp.mcp_service import (
    MAX_PUBLIC_NAME_LENGTH,
    McpClient,
    McpToolDefinition,
    mcp_tool_to_spec,
    normalize_mcp_tool_name,
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


def test_mcp_tool_spec_json_roundtrip_preserves_rebuild_config() -> None:
    config = McpServerConfig(name="tools", url="https://mcp.example.com")
    spec = mcp_tool_to_spec(config, _remote_tool_def())

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
    spec = mcp_tool_to_spec(config, _remote_tool_def())
    client = MagicMock()
    client.list_tools = AsyncMock(return_value=[_remote_tool_def()])
    client.call_tool = AsyncMock(return_value="result:platform")
    client.close = AsyncMock()

    with patch("private_gpt.server.mcp.mcp_service.McpClient", return_value=client):
        rebuilt = await rebuild_tool_from_spec(spec)
        result = await rebuilt.acall(query="platform")

    assert result.raw_output == "result:platform"
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
    # The public name is collision-safe normalized and namespaced by server.
    assert tools[0].name == normalize_mcp_tool_name("Test MCP", "Call_test_tool_object_")
    # The raw MCP name is preserved for the wire.
    assert tools[0].raw_name == "Call_test_tool_object_"
    assert tools[0].input_schema == original_schema
    assert "type" not in tools[0].input_schema["properties"]["object_param"]


# ---------------------------------------------------------------------------
# Name normalization (mirrors the deepseek-harness mcp-client contract)
# ---------------------------------------------------------------------------


def test_normalize_clean_name_is_verbatim() -> None:
    assert (
        normalize_mcp_tool_name("files", "read_file")
        == "mcp__files__read_file"
    )


def test_normalize_replaces_invalid_chars_and_hashes() -> None:
    public = normalize_mcp_tool_name("Test MCP", "Call_test_tool_object_")
    assert public.startswith("mcp__Test_MCP__Call_test_tool_object_")
    # A 12-hex SHA-256 suffix is appended when normalization is lossy.
    assert len(public) <= MAX_PUBLIC_NAME_LENGTH
    assert "_" in public
    suffix = public.split("_")[-1]
    assert len(suffix) == 12
    assert all(c in "0123456789abcdef" for c in suffix)


def test_normalize_truncates_overlong_names_with_hash() -> None:
    raw = "very_long_tool_" + "x" * 80
    public = normalize_mcp_tool_name("srv", raw)
    assert len(public) <= MAX_PUBLIC_NAME_LENGTH
    assert "_" in public
    suffix = public.split("_")[-1]
    assert len(suffix) == 12


def test_normalize_two_colliding_names_stay_distinct() -> None:
    # After replacement both become the same prefix, so the hash must
    # disambiguate them.
    first = normalize_mcp_tool_name("srv", "a.b")
    second = normalize_mcp_tool_name("srv", "a b")
    assert first != second
    assert first.startswith("mcp__srv__a_b")
    assert second.startswith("mcp__srv__a_b")


def test_normalize_is_deterministic() -> None:
    assert normalize_mcp_tool_name("srv", "a.b") == normalize_mcp_tool_name(
        "srv", "a.b"
    )
    assert normalize_mcp_tool_name("srv", "a.b") != normalize_mcp_tool_name(
        "other", "a.b"
    )


@pytest.mark.asyncio
async def test_call_tool_resolves_public_name_to_raw_name() -> None:
    config = McpServerConfig(name="srv", url="https://mcp.example.com")
    raw_tool = Tool(
        name="Call_test_tool_object_",
        description="d",
        inputSchema={"type": "object", "properties": {}},
    )
    runtime_client = MagicMock()
    runtime_client.list_tools = AsyncMock(
        return_value=ListToolsResult(tools=[raw_tool])
    )
    runtime_client.call_tool = AsyncMock(return_value="ok")
    with patch(
        "private_gpt.server.mcp.mcp_service._load_runtime",
        return_value=MagicMock(return_value=runtime_client),
    ):
        client = McpClient(config)
        await client.list_tools()
        public_name = normalize_mcp_tool_name("srv", "Call_test_tool_object_")
        await client.call_tool(public_name, {"x": 1})

    runtime_client.call_tool.assert_awaited_once_with(
        name="Call_test_tool_object_", arguments={"x": 1}
    )
