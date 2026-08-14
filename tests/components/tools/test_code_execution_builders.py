from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from private_gpt.components.code_execution.results import (
    BashExecutionResult,
    FileOperationResult,
)
from private_gpt.components.tools.builders.bash_tool_builder import BashToolBuilder
from private_gpt.components.tools.builders.present_files_tool_builder import (
    PresentFilesToolBuilder,
)
from private_gpt.components.tools.builders.text_editor_tool_builder import (
    TextEditorToolBuilder,
)
from private_gpt.components.tools.events.adapters import (
    BashCodeExecutionEventAdapter,
    PresentFilesEventAdapter,
    TextEditorCodeExecutionEventAdapter,
)
from private_gpt.settings.settings import unsafe_typed_settings


def _settings():
    settings = unsafe_typed_settings.model_copy(deep=True)
    settings.code_execution.max_output_bytes = 10_000
    return settings


@pytest.mark.asyncio
async def test_bash_tool_builder_executes_session_command() -> None:
    session = SimpleNamespace(
        execute_bash=AsyncMock(
            return_value=BashExecutionResult(
                success=True,
                stdout="ok",
                stderr="",
                exit_code=0,
            )
        )
    )
    builder = BashToolBuilder(
        code_execution_component=SimpleNamespace(
            get_or_create_session=AsyncMock(return_value=session)
        ),
        settings=_settings(),
    )

    tool = await builder.build_tool("corr-1")
    result = await tool.async_fn(command="echo ok")

    session.execute_bash.assert_awaited_once_with(
        "echo ok",
        timeout=None,
        restart=False,
    )
    assert result[0].type == "bash_code_execution_result"
    assert result[0].stdout == "ok"
    assert result[0].stderr == ""
    assert result[0].return_code == 0
    assert tool.event_adapter is BashCodeExecutionEventAdapter


@pytest.mark.asyncio
async def test_text_editor_tool_builder_wraps_file_operations() -> None:
    session = SimpleNamespace(
        view=AsyncMock(
            return_value=FileOperationResult(success=True, output="1: line")
        ),
        str_replace=AsyncMock(
            return_value=FileOperationResult(success=True, output="Updated file.txt")
        ),
        create=AsyncMock(return_value=FileOperationResult(success=True)),
        insert=AsyncMock(
            return_value=FileOperationResult(success=True, output="Updated file.txt")
        ),
    )
    builder = TextEditorToolBuilder(
        code_execution_component=SimpleNamespace(
            get_or_create_session=AsyncMock(return_value=session)
        ),
        settings=_settings(),
    )

    view_tool = await builder.build_view_tool("corr-2")
    replace_tool = await builder.build_str_replace_tool("corr-2")
    create_tool = await builder.build_create_tool("corr-2")
    insert_tool = await builder.build_insert_tool("corr-2")

    view_result = await view_tool.async_fn(path="file.txt", view_range=[1, 1])
    replace_result = await replace_tool.async_fn(
        path="file.txt",
        old_str="old",
        new_str="new",
    )
    create_result = await create_tool.async_fn(path="file.txt", file_text="body")
    insert_result = await insert_tool.async_fn(
        path="file.txt",
        insert_line=1,
        new_str="extra",
    )

    session.view.assert_awaited_once_with("file.txt", view_range=(1, 1))
    session.str_replace.assert_awaited_once_with("file.txt", "old", "new")
    session.create.assert_awaited_once_with("file.txt", "body")
    session.insert.assert_awaited_once_with("file.txt", 1, "extra")
    assert view_result[0].type == "text_editor_code_execution_view_result"
    assert view_result[0].content == "1: line"
    assert replace_result[0].type == "text_editor_code_execution_str_replace_result"
    assert replace_result[0].lines == ["Updated file.txt"]
    assert create_result[0].type == "text_editor_code_execution_create_result"
    assert not create_result[0].is_file_update
    assert insert_result[0].type == "text_editor_code_execution_str_replace_result"
    assert insert_result[0].new_lines == 1
    assert view_tool.event_adapter is TextEditorCodeExecutionEventAdapter


@pytest.mark.asyncio
async def test_text_editor_view_puts_no_output_in_empty_file_content() -> None:
    session = SimpleNamespace(
        view=AsyncMock(return_value=FileOperationResult(success=True, output=""))
    )
    builder = TextEditorToolBuilder(
        code_execution_component=SimpleNamespace(
            get_or_create_session=AsyncMock(return_value=session)
        ),
        settings=_settings(),
    )

    view_tool = await builder.build_view_tool("corr-empty")
    view_result = await view_tool.async_fn(path="empty.txt")

    assert len(view_result) == 1
    assert view_result[0].type == "text_editor_code_execution_view_result"
    assert view_result[0].content == "(no-output)"
    assert view_result[0].num_lines == 0
    assert view_result[0].total_lines == 0


@pytest.mark.asyncio
async def test_present_files_builder_presents_existing_files() -> None:
    session = SimpleNamespace(path_exists=AsyncMock(return_value=True))
    builder = PresentFilesToolBuilder(
        code_execution_component=SimpleNamespace(
            get_or_create_session=AsyncMock(return_value=session)
        )
    )

    tool = await builder.build_tool("corr-present")
    result = await tool.async_fn(
        filepaths=["/mnt/user-data/outputs/chart.png", "/tmp/notes.md"]
    )

    assert session.path_exists.await_count == 2
    session.path_exists.assert_any_await("/mnt/user-data/outputs/chart.png")
    session.path_exists.assert_any_await("/tmp/notes.md")
    assert [block.type for block in result] == [
        "local_resource",
        "local_resource",
        "text",
    ]
    assert result[0].file_path == "/mnt/user-data/outputs/chart.png"
    assert result[0].name == "chart"
    assert result[0].mime_type == "image/png"
    assert result[1].file_path == "/tmp/notes.md"
    assert result[1].mime_type == "text/markdown"
    assert result[2].text == "Presented 2 file(s): chart.png, notes.md"
    assert tool.event_adapter is PresentFilesEventAdapter


@pytest.mark.asyncio
async def test_present_files_builder_returns_error_when_file_missing() -> None:
    session = SimpleNamespace(path_exists=AsyncMock(return_value=False))
    builder = PresentFilesToolBuilder(
        code_execution_component=SimpleNamespace(
            get_or_create_session=AsyncMock(return_value=session)
        )
    )

    tool = await builder.build_tool("corr-missing")
    result = await tool.async_fn(filepaths=["/mnt/user-data/outputs/missing.png"])

    session.path_exists.assert_awaited_once_with("/mnt/user-data/outputs/missing.png")
    assert [block.type for block in result] == ["text", "text"]
    assert (
        result[0].text
        == "Error presenting /mnt/user-data/outputs/missing.png: File not found: /mnt/user-data/outputs/missing.png"
    )
    assert result[1].text == "No files could be presented."
