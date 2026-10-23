from llama_index.core.llms.llm import ToolSelection
from openai.types.chat.chat_completion_chunk import (
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from private_gpt.components.engines.chat.utils.tool_utils import merge_stream_tool_calls


def test_merge_stream_tool_calls_replaces_tool_selection_by_id() -> None:
    existing = [
        ToolSelection(tool_id="tool_1", tool_name="echo", tool_kwargs={"value": "a"}),
    ]
    incoming = [
        ToolSelection(tool_id="tool_1", tool_name="echo", tool_kwargs={"value": "b"}),
        ToolSelection(tool_id="tool_2", tool_name="echo", tool_kwargs={"value": "c"}),
    ]

    merged = merge_stream_tool_calls(existing, incoming)

    assert [tc.tool_id for tc in merged] == ["tool_1", "tool_2"]
    assert merged[0].tool_kwargs == {"value": "b"}
    assert merged[1].tool_kwargs == {"value": "c"}


def test_merge_stream_tool_calls_replaces_openai_snapshots_by_id() -> None:
    existing = [
        ChoiceDeltaToolCall(
            index=0,
            id="call_abc",
            type="function",
            function=ChoiceDeltaToolCallFunction(name="echo", arguments="{"),
        )
    ]
    incoming = [
        ChoiceDeltaToolCall(
            index=0,
            id="call_abc",
            type="function",
            function=ChoiceDeltaToolCallFunction(
                name="echo", arguments='{"value": "x"}'
            ),
        )
    ]

    merged = merge_stream_tool_calls(existing, incoming)

    assert len(merged) == 1
    assert merged[0].id == "call_abc"
    assert merged[0].function.arguments == '{"value": "x"}'


def test_merge_stream_tool_calls_accumulates_openai_argument_deltas() -> None:
    existing = [
        ChoiceDeltaToolCall(
            index=0,
            id="call_abc",
            type="function",
            function=ChoiceDeltaToolCallFunction(name="echo", arguments="{"),
        )
    ]
    incoming = [
        ChoiceDeltaToolCall(
            index=0,
            function=ChoiceDeltaToolCallFunction(arguments='"value": "x"}'),
        )
    ]

    merged = merge_stream_tool_calls(existing, incoming)

    assert len(merged) == 1
    assert merged[0].id == "call_abc"
    assert merged[0].function.name == "echo"
    assert merged[0].function.arguments == '{"value": "x"}'
    assert merged[0].type == "function"


def test_merge_stream_tool_calls_does_not_read_missing_tool_id() -> None:
    incoming = [
        ChoiceDeltaToolCall(
            index=0,
            id="call_abc",
            type="function",
            function=ChoiceDeltaToolCallFunction(name="echo", arguments="{}"),
        )
    ]

    merged = merge_stream_tool_calls([], incoming)

    assert len(merged) == 1
    assert merged[0].id == "call_abc"
    assert not hasattr(merged[0], "tool_id")
