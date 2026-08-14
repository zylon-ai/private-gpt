from types import SimpleNamespace

import pytest
from llama_index.core.tools import FunctionTool

from private_gpt.components.engines.chat.utils.tool_utils import execute_tool_call
from private_gpt.events.models import (
    BashCodeExecutionResultBlock,
    TextEditorCodeExecutionViewResultBlock,
)


@pytest.mark.anyio
@pytest.mark.parametrize("value", [None, [], "", "   "])
async def test_execute_tool_call_keeps_empty_result_visible(value: object) -> None:
    async def noop() -> object:
        return value

    tool = FunctionTool.from_defaults(
        async_fn=noop,
        name="noop",
        description="Returns no content",
    )

    result, message = await execute_tool_call(
        tool=tool,
        tool_name="noop",
        tool_id="tc_empty",
        tool_kwargs={},
        state_ctx=SimpleNamespace(),
    )

    assert result.tool_output.content == "(no-output)"
    assert message.content == "(no-output)"
    assert message.additional_kwargs["tool_call_id"] == "tc_empty"


@pytest.mark.anyio
async def test_execute_tool_call_renders_bash_stdout() -> None:
    result_block = BashCodeExecutionResultBlock(
        stdout="# Potato\n",
        stderr="",
        return_code=0,
    )

    async def run_bash() -> list[BashCodeExecutionResultBlock]:
        return [result_block]

    tool = FunctionTool.from_defaults(
        async_fn=run_bash,
        name="bash_code_execution",
        description="Run a bash command",
    )

    result, message = await execute_tool_call(
        tool=tool,
        tool_name="bash_code_execution",
        tool_id="srvtoolu_bash",
        tool_kwargs={},
        state_ctx=SimpleNamespace(),
    )

    assert result.tool_output.content == result_block.render()
    assert message.content == result_block.render()
    assert "# Potato" in (message.content or "")
    assert message.content != "(no-output)"
    assert "bash_code_execution_result" in message.additional_kwargs


@pytest.mark.anyio
async def test_execute_tool_call_renders_text_editor_view() -> None:
    view = TextEditorCodeExecutionViewResultBlock(
        content="# Potato (Solanum tuberosum)\n",
        num_lines=1,
        start_line=1,
        total_lines=1,
    )

    async def view_file() -> list[TextEditorCodeExecutionViewResultBlock]:
        return [view]

    tool = FunctionTool.from_defaults(
        async_fn=view_file,
        name="text_editor_code_execution",
        description="View a file",
    )

    result, message = await execute_tool_call(
        tool=tool,
        tool_name="text_editor_code_execution",
        tool_id="srvtoolu_view",
        tool_kwargs={},
        state_ctx=SimpleNamespace(),
    )

    assert result.tool_output.content == view.content
    assert message.content == view.content
    assert message.content != "(no-output)"
