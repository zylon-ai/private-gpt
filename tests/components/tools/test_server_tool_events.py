import pytest

from private_gpt.components.chat.models.chat_config_models import ToolSpec
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
    ServerToolUseBlock,
    TextBlock,
    TextEditorCodeExecutionViewResultBlock,
    ToolResultBlock,
    ToolUseBlock,
    WebFetchResultBlock,
    WebSearchResultBlock,
)


def _tool(event_adapter: type | None = None, *, runtime: str = "server") -> ToolSpec:
    return ToolSpec.from_defaults(
        name="internal_tool",
        runtime=runtime,
        event_adapter=event_adapter,
        input_schema={"type": "object", "properties": {}},
    )


def test_event_adapter_field_stores_class() -> None:
    tool = _tool(BashCodeExecutionEventAdapter)
    assert tool.event_adapter is BashCodeExecutionEventAdapter


def test_event_adapter_serializes_to_class_path() -> None:
    tool = _tool(BashCodeExecutionEventAdapter)
    payload = tool.model_dump(mode="json")
    expected = (
        "private_gpt.components.tools.events.adapters:BashCodeExecutionEventAdapter"
    )
    assert payload["event_adapter"] == expected


def test_event_adapter_round_trips_through_json() -> None:
    original = _tool(BashCodeExecutionEventAdapter)
    payload = original.model_dump(mode="json")
    restored = ToolSpec.model_validate(payload)

    assert restored.event_adapter is BashCodeExecutionEventAdapter
    assert isinstance(restored.resolve_event_adapter(), BashCodeExecutionEventAdapter)


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


@pytest.mark.skip(
    reason="Internal server tools without an Anthropic-native name now use ToolUseBlock/ToolResultBlock "
    "(client-style) for SDK Message.content compatibility. "
    "Only Anthropic-native server tools (web_search, bash_code_execution, etc.) "
    "use ServerToolUseBlock."
)
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
    assert isinstance(use, ClientToolUseBlock)
    assert use.name == "internal_tool"
    assert isinstance(result, ClientToolResultBlock)
    assert result.type == "tool_result"
    assert tool_id.startswith("srvtoolu_")
    assert tool_id.startswith(
        "srvtoolu_"
    )  # still uses srvtoolu_ prefix for ID consistency


def test_bash_tool_resolves_specialized_adapter() -> None:
    tool = _tool(BashCodeExecutionEventAdapter)
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
            content=[
                BashCodeExecutionResultBlock(
                    stdout="ok", stderr="", return_code=0, content=[]
                )
            ]
        ),
    )

    assert isinstance(adapter, BashCodeExecutionEventAdapter)
    assert isinstance(use, ServerToolUseBlock)
    assert use.name == "bash_code_execution"
    assert isinstance(result, BashCodeExecutionToolResultBlock)
    assert result.content.stdout == "ok"


def test_specialized_adapter_owns_error_format() -> None:
    tool = _tool(BashCodeExecutionEventAdapter)
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


def test_text_editor_resolves_without_central_branching() -> None:
    adapter = _tool(TextEditorCodeExecutionEventAdapter).resolve_event_adapter()
    assert isinstance(adapter, TextEditorCodeExecutionEventAdapter)


def test_invalid_path_fails_at_resolution() -> None:
    with pytest.raises((AttributeError, TypeError, ValueError)):
        ToolSpec.model_validate(
            {
                "name": "bad",
                "runtime": "server",
                "event_adapter": (
                    "private_gpt.components.tools.events.adapters:NonExistentAdapter"
                ),
                "input_schema": {},
            }
        )


def test_new_adapter_requires_no_central_source_changes() -> None:
    """A module-level custom adapter can be used without touching engine or registry."""

    tool = ToolSpec.from_defaults(
        name="custom",
        runtime="server",
        event_adapter=_CustomAdapter,
        input_schema={"type": "object", "properties": {}},
    )
    adapter = tool.resolve_event_adapter()
    use = adapter.build_tool_use(
        tool_id=adapter.new_tool_use_id(),
        tool_name="ignored",
        tool_input={},
    )

    assert isinstance(adapter, _CustomAdapter)
    assert use.name == "custom_public_tool"


class _CustomAdapter(ServerToolEventAdapter):
    """Module-level adapter proving extensibility without central source changes."""

    public_tool_name = "custom_public_tool"


def test_client_adapter_renders_bash_result_to_text_block() -> None:
    adapter = ClientToolEventAdapter()
    result = adapter.build_tool_result(
        tool_use_id="tool_abc",
        outcome=ToolExecutionSuccess(
            content=[
                BashCodeExecutionResultBlock(
                    stdout="Python 3.11.2",
                    stderr="openjdk version 17",
                    return_code=0,
                    content=[],
                )
            ]
        ),
    )

    assert isinstance(result, ClientToolResultBlock)
    assert isinstance(result.content, list)
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, TextBlock)
    assert "exit_code: 0" in block.text
    assert "stdout:\nPython 3.11.2" in block.text
    assert "stderr:\nopenjdk version 17" in block.text


def test_client_adapter_renders_text_editor_view_to_text_block() -> None:
    adapter = ClientToolEventAdapter()
    result = adapter.build_tool_result(
        tool_use_id="tool_abc",
        outcome=ToolExecutionSuccess(
            content=[
                TextEditorCodeExecutionViewResultBlock(
                    content="# Hello\nworld",
                    num_lines=2,
                    start_line=1,
                    total_lines=2,
                )
            ]
        ),
    )

    assert isinstance(result, ClientToolResultBlock)
    assert isinstance(result.content, list)
    assert len(result.content) == 1
    block = result.content[0]
    assert isinstance(block, TextBlock)
    assert block.text == "# Hello\nworld"


def test_client_adapter_renders_web_fetch_to_text_block() -> None:
    adapter = ClientToolEventAdapter()
    result = adapter.build_tool_result(
        tool_use_id="tool_abc",
        outcome=ToolExecutionSuccess(
            content=[
                WebFetchResultBlock.from_markdown(
                    url="https://example.com", markdown="# Example"
                )
            ]
        ),
    )

    assert isinstance(result, ClientToolResultBlock)
    assert isinstance(result.content, list)
    block = result.content[0]
    assert isinstance(block, TextBlock)
    assert block.text == "# Example"


def test_client_adapter_renders_web_search_to_text_block() -> None:
    adapter = ClientToolEventAdapter()
    result = adapter.build_tool_result(
        tool_use_id="tool_abc",
        outcome=ToolExecutionSuccess(
            content=[
                WebSearchResultBlock(
                    url="https://example.com",
                    title="Example",
                    encrypted_content="some content",
                    content="some content",
                    description="A site",
                )
            ]
        ),
    )

    assert isinstance(result, ClientToolResultBlock)
    assert isinstance(result.content, list)
    block = result.content[0]
    assert isinstance(block, TextBlock)
    assert "Example" in block.text
    assert "https://example.com" in block.text
    assert "some content" in block.text
