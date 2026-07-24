from typing import cast
from uuid import uuid4

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.events.models import (
    BashCodeExecutionResultBlock,
    BashCodeExecutionToolResultBlock,
    ClientToolResultBlock,
    ClientToolUseBlock,
    CodeExecutionToolResultErrorBlock,
    ContentBlockType,
    ResultContentBlockType,
    ServerToolUseBlock,
    TextEditorCodeExecutionCreateResultBlock,
    TextEditorCodeExecutionStrReplaceResultBlock,
    TextEditorCodeExecutionToolResultBlock,
    TextEditorCodeExecutionViewResultBlock,
)


def new_tool_use_id(tool_spec: ToolSpec | None) -> str:
    prefix = "srvtoolu" if tool_spec and tool_spec.server_tool_name else "tool"
    return f"{prefix}_{uuid4().hex}"


def build_tool_use_block(
    tool_spec: ToolSpec | None,
    *,
    tool_id: str,
    tool_name: str,
    tool_input: dict,
) -> ContentBlockType:
    if tool_spec and tool_spec.server_tool_name:
        return ServerToolUseBlock(
            id=tool_id,
            name=tool_spec.server_tool_name,
            input=tool_input,
        )
    return ClientToolUseBlock(id=tool_id, name=tool_name, input=tool_input)


def build_tool_result_block(
    tool_spec: ToolSpec | None,
    *,
    tool_use_id: str,
    content: str | list[ResultContentBlockType],
    is_error: bool,
) -> ContentBlockType:
    if not tool_spec or not tool_spec.server_tool_name:
        return ClientToolResultBlock(
            tool_use_id=tool_use_id,
            content=content,
            is_error=is_error,
        )

    if is_error:
        error_type = (
            "bash_code_execution_tool_result_error"
            if tool_spec.server_tool_name == "bash_code_execution"
            else "text_editor_code_execution_tool_result_error"
        )
        error = CodeExecutionToolResultErrorBlock(
            type=error_type,
            error_code="unavailable",
        )
        if tool_spec.server_tool_name == "bash_code_execution":
            return BashCodeExecutionToolResultBlock(
                tool_use_id=tool_use_id,
                content=error,
            )
        return TextEditorCodeExecutionToolResultBlock(
            tool_use_id=tool_use_id,
            content=error,
        )

    if not isinstance(content, list) or len(content) != 1:
        raise ValueError("Server tools must return exactly one structured result block")
    result = content[0]
    if tool_spec.server_tool_name == "bash_code_execution":
        return BashCodeExecutionToolResultBlock(
            tool_use_id=tool_use_id,
            content=cast(BashCodeExecutionResultBlock, result),
        )
    return TextEditorCodeExecutionToolResultBlock(
        tool_use_id=tool_use_id,
        content=cast(
            TextEditorCodeExecutionViewResultBlock
            | TextEditorCodeExecutionCreateResultBlock
            | TextEditorCodeExecutionStrReplaceResultBlock,
            result,
        ),
    )
