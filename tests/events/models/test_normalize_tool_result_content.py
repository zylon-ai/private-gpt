from private_gpt.events.models import (
    NO_TOOL_CONTENT,
    TextBlock,
    TextEditorCodeExecutionViewResultBlock,
    normalize_tool_result_content,
)


def test_normalize_empty_list_gets_placeholder() -> None:
    assert normalize_tool_result_content([]) == [TextBlock(text=NO_TOOL_CONTENT)]


def test_normalize_fills_empty_text_block_in_place() -> None:
    normalized = normalize_tool_result_content([TextBlock(text="   ")])

    assert normalized == [TextBlock(text=NO_TOOL_CONTENT)]


def test_normalize_appends_placeholder_when_existing_blocks_have_no_text() -> None:
    view = TextEditorCodeExecutionViewResultBlock(
        content="",
        num_lines=0,
        start_line=1,
        total_lines=0,
    )

    normalized = normalize_tool_result_content([view])

    assert normalized[0] is view
    assert normalized[1] == TextBlock(text=NO_TOOL_CONTENT)
