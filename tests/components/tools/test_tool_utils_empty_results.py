from types import SimpleNamespace

import pytest
from llama_index.core.tools import FunctionTool

from private_gpt.components.engines.chat.utils.tool_utils import execute_tool_call


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
