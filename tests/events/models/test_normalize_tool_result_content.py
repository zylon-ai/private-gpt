from llama_index.core.base.llms.types import TextBlock as LITextBlock

from private_gpt.events.models import (
    NO_TOOL_CONTENT,
    BashCodeExecutionResultBlock,
    TextBlock,
    TextEditorCodeExecutionViewResultBlock,
    normalize_tool_result_content,
    to_llama_index_blocks,
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


def test_to_llama_index_blocks_renders_bash_stdout() -> None:
    result = BashCodeExecutionResultBlock(
        stdout="# Potato\n",
        stderr="",
        return_code=0,
    )

    blocks = to_llama_index_blocks([result])

    assert len(blocks) == 1
    assert isinstance(blocks[0], LITextBlock)
    assert blocks[0].text == result.render()
    assert "# Potato" in blocks[0].text


def test_to_llama_index_blocks_renders_text_editor_view() -> None:
    view = TextEditorCodeExecutionViewResultBlock(
        content="# Potato (Solanum tuberosum)\n",
        num_lines=1,
        start_line=1,
        total_lines=1,
    )

    blocks = to_llama_index_blocks([view])

    assert len(blocks) == 1
    assert isinstance(blocks[0], LITextBlock)
    assert blocks[0].text == view.content
