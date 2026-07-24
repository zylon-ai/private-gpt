import pytest

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.tools.events import register_tool_event_adapter
from private_gpt.components.tools.events.adapters import (
    BashCodeExecutionEventAdapter,
    ClientToolEventAdapter,
    ServerToolEventAdapter,
    TextEditorCodeExecutionEventAdapter,
)
from private_gpt.components.tools.tool_execution_outcome import (
    ToolExecutionError,
    ToolExecutionFailure,
    ToolExecutionSuccess,
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


def _tool(event_adapter_key: str | None = None, *, runtime: str = "server") -> ToolSpec:
    return ToolSpec.from_defaults(
        name="internal_tool",
        runtime=runtime,
        event_adapter_key=event_adapter_key,
        input_schema={"type": "object", "properties": {}},
    )


def test_client_tool_resolves_default_client_adapter() -> None:
    tool = _tool(runtime="client")
    adapter = tool.resolve_event_adapter()
    tool_id = adapter.new_tool_use_id()
    use = adapter.build_tool_use(
        tool_id=tool_id, tool_name="internal_tool", tool_input={}
    )
    result = adapter.build_tool_result(
        tool_use_id=tool_id,
        outcome=ToolExecutionSuccess(content=[]),
    )

    assert isinstance(adapter, ClientToolEventAdapter)
    assert isinstance(use, ClientToolUseBlock)
    assert isinstance(use, ToolUseBlock)
    assert isinstance(result, ClientToolResultBlock)
    assert isinstance(result, ToolResultBlock)
    assert tool_id.startswith("tool_")


def test_server_tool_resolves_default_server_adapter() -> None:
    tool = _tool()
    adapter = tool.resolve_event_adapter()
    tool_id = adapter.new_tool_use_id()
    use = adapter.build_tool_use(
        tool_id=tool_id,
        tool_name="semantic_search",
        tool_input={"query": "test"},
    )
    result = adapter.build_tool_result(
        tool_use_id=tool_id,
        outcome=ToolExecutionSuccess(content=[]),
    )

    assert isinstance(adapter, ServerToolEventAdapter)
    assert isinstance(use, ServerToolUseBlock)
    assert use.name == "semantic_search"
    assert isinstance(result, ServerToolResultBlock)
    assert result.type == "server_tool_result"
    assert tool_id.startswith("srvtoolu_")


def test_bash_tool_resolves_specialized_adapter() -> None:
    tool = _tool("code_execution.bash")
    adapter = tool.resolve_event_adapter()
    tool_id = adapter.new_tool_use_id()
    use = adapter.build_tool_use(
        tool_id=tool_id,
        tool_name="bash",
        tool_input={"command": "echo ok"},
    )
    result = adapter.build_tool_result(
        tool_use_id=tool_id,
        outcome=ToolExecutionSuccess(
            content=[BashCodeExecutionResultBlock(stdout="ok")]
        ),
    )

    assert isinstance(adapter, BashCodeExecutionEventAdapter)
    assert isinstance(use, ServerToolUseBlock)
    assert isinstance(result, BashCodeExecutionToolResultBlock)
    assert use.name == "bash_code_execution"
    assert result.content.stdout == "ok"


def test_specialized_adapter_owns_error_format() -> None:
    tool = _tool("code_execution.bash")
    adapter = tool.resolve_event_adapter()
    result = adapter.build_tool_result(
        tool_use_id="srvtoolu_test",
        outcome=ToolExecutionFailure(
            error=ToolExecutionError(message="sandbox unavailable")
        ),
    )

    assert isinstance(result, BashCodeExecutionToolResultBlock)
    assert result.content.type == "bash_code_execution_tool_result_error"
    assert result.content.error_code == "unavailable"


def test_text_editor_key_resolves_without_shared_logic_branching() -> None:
    adapter = _tool("code_execution.text_editor").resolve_event_adapter()
    assert isinstance(adapter, TextEditorCodeExecutionEventAdapter)


def test_unknown_adapter_key_fails_at_resolution() -> None:
    tool = _tool("missing.adapter")
    with pytest.raises(ValueError, match=r"missing\.adapter"):
        tool.resolve_event_adapter()


def test_new_adapter_can_be_registered_without_changing_resolution_logic() -> None:
    class CustomAdapter(ServerToolEventAdapter):
        public_tool_name = "custom_public_tool"

    register_tool_event_adapter("test.custom", CustomAdapter())
    tool = _tool("test.custom")
    adapter = tool.resolve_event_adapter()
    use = adapter.build_tool_use(
        tool_id=adapter.new_tool_use_id(),
        tool_name="internal_name",
        tool_input={},
    )

    assert isinstance(adapter, CustomAdapter)
    assert use.name == "custom_public_tool"
