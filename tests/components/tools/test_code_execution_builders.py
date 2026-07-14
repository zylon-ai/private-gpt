from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from private_gpt.components.code_execution.results import (
    BashExecutionResult,
    FileOperationResult,
)
from private_gpt.components.tools.builders.bash_tool_builder import BashToolBuilder
from private_gpt.components.tools.builders.text_editor_tool_builder import (
    TextEditorToolBuilder,
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
    assert result[0].text == "exit_code: 0\n\nstdout:\nok"


@pytest.mark.asyncio
async def test_text_editor_tool_builder_wraps_file_operations() -> None:
    session = SimpleNamespace(
        view=AsyncMock(
            return_value=FileOperationResult(success=True, output="1: line")
        ),
        str_replace=AsyncMock(
            return_value=FileOperationResult(success=True, output="Updated file.txt")
        ),
        create=AsyncMock(
            return_value=FileOperationResult(success=False, error="exists")
        ),
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
    assert view_result[0].text == "1: line"
    assert replace_result[0].text == "Updated file.txt"
    assert create_result[0].text == "Error: exists"
    assert insert_result[0].text == "Updated file.txt"
