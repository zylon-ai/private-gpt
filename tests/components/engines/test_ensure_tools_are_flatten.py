import pytest
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms.llm import ToolSelection

from private_gpt.components.engines.chat.interceptors.ensure_tools_are_flatten_interceptor import (
    EnsureToolAreFlattenInterceptor,
)


def _call(tool_id: str) -> ToolSelection:
    return ToolSelection(tool_id=tool_id, tool_name="bash", tool_kwargs={"c": tool_id})


@pytest.mark.parametrize("calls", [1, 2])
def test_flattening_keeps_the_reasoning_on_the_first_call(calls: int) -> None:
    ids = [f"t{i}" for i in range(calls)]
    messages = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(
            role="assistant",
            content="",
            additional_kwargs={
                "tool_calls": [_call(i) for i in ids],
                "thinking": "Need the shell.",
                "token_ids_delta": [1, 2],
            },
        ),
        *[
            ChatMessage(
                role="tool", content="ok", additional_kwargs={"tool_call_id": i}
            )
            for i in ids
        ],
    ]

    flat = EnsureToolAreFlattenInterceptor()._reorder_tool_messages(messages)

    assistants = [m for m in flat if m.role == "assistant"]
    assert len(assistants) == calls
    assert assistants[0].additional_kwargs["thinking"] == "Need the shell."
    assert [a.additional_kwargs["tool_calls"][0].tool_id for a in assistants] == ids
    for later in assistants[1:]:
        assert "thinking" not in later.additional_kwargs
    # Other streaming leftovers are still dropped.
    assert all("token_ids_delta" not in a.additional_kwargs for a in assistants)
