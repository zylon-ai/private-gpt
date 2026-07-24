from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.tools.server_tool_events import (
    build_tool_result_block,
    build_tool_use_block,
    new_tool_use_id,
)
from private_gpt.events.models import (
    BashCodeExecutionResultBlock,
    BashCodeExecutionToolResultBlock,
    ClientToolResultBlock,
    ClientToolUseBlock,
    ServerToolResultBlock,
    ServerToolUseBlock,
    ToolResultBlock,
    ToolUseBlock,
)


def _tool(server_tool_name: str | None = None, *, runtime: str = "server") -> ToolSpec:
    return ToolSpec.from_defaults(
        name="internal_tool",
        runtime=runtime,
        server_tool_name=server_tool_name,
        input_schema={"type": "object", "properties": {}},
    )


def test_client_blocks_implement_shared_interfaces() -> None:
    tool = _tool(runtime="client")
    tool_id = new_tool_use_id(tool)
    use = build_tool_use_block(
        tool, tool_id=tool_id, tool_name="internal_tool", tool_input={}
    )
    result = build_tool_result_block(
        tool, tool_use_id=tool_id, content="ok", is_error=False
    )

    assert isinstance(use, ClientToolUseBlock)
    assert isinstance(use, ToolUseBlock)
    assert isinstance(result, ClientToolResultBlock)
    assert isinstance(result, ToolResultBlock)
    assert tool_id.startswith("tool_")


def test_generic_server_blocks_use_default_server_result() -> None:
    tool = _tool()
    tool_id = new_tool_use_id(tool)
    use = build_tool_use_block(
        tool,
        tool_id=tool_id,
        tool_name="semantic_search",
        tool_input={"query": "test"},
    )
    result = build_tool_result_block(
        tool,
        tool_use_id=tool_id,
        content="found",
        is_error=False,
    )

    assert isinstance(use, ServerToolUseBlock)
    assert use.name == "semantic_search"
    assert isinstance(result, ServerToolResultBlock)
    assert result.type == "server_tool_result"
    assert result.content == "found"
    assert tool_id.startswith("srvtoolu_")


def test_bash_server_blocks_implement_shared_interfaces() -> None:
    tool = _tool("bash_code_execution")
    tool_id = new_tool_use_id(tool)
    use = build_tool_use_block(
        tool,
        tool_id=tool_id,
        tool_name="bash",
        tool_input={"command": "echo ok"},
    )
    result = build_tool_result_block(
        tool,
        tool_use_id=tool_id,
        content=[BashCodeExecutionResultBlock(stdout="ok")],
        is_error=False,
    )

    assert isinstance(use, ServerToolUseBlock)
    assert isinstance(use, ToolUseBlock)
    assert isinstance(result, BashCodeExecutionToolResultBlock)
    assert isinstance(result, ToolResultBlock)
    assert use.name == "bash_code_execution"
    assert tool_id.startswith("srvtoolu_")
    assert result.model_dump() == {
        "type": "bash_code_execution_tool_result",
        "tool_use_id": tool_id,
        "content": {
            "type": "bash_code_execution_result",
            "stdout": "ok",
            "stderr": "",
            "return_code": 0,
            "content": [],
        },
    }
