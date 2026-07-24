from __future__ import annotations

from typing import TYPE_CHECKING, cast
from uuid import uuid4

from private_gpt.components.tools.tool_execution_outcome import (
    ToolExecutionFailure,
)
from private_gpt.events.models import (
    BashCodeExecutionResultBlock,
    BashCodeExecutionToolResultBlock,
    ClientToolResultBlock,
    ClientToolUseBlock,
    CodeExecutionToolResultErrorBlock,
    ServerToolResultBlock,
    ServerToolUseBlock,
    TextEditorCodeExecutionCreateResultBlock,
    TextEditorCodeExecutionStrReplaceResultBlock,
    TextEditorCodeExecutionToolResultBlock,
    TextEditorCodeExecutionViewResultBlock,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from private_gpt.components.tools.tool_execution_outcome import ToolExecutionOutcome
    from private_gpt.events.models import (
        ResultContentBlockType,
        ToolResultBlock,
        ToolUseBlock,
    )


class ToolEventAdapter:
    id_prefix = "tool"
    public_tool_name: str | None = None

    def __init__(self, id_factory: Callable[[], str] | None = None) -> None:
        self._id_factory = id_factory or (lambda: uuid4().hex)

    def new_tool_use_id(self) -> str:
        return f"{self.id_prefix}_{self._id_factory()}"

    def build_tool_use(
        self,
        *,
        tool_id: str,
        tool_name: str,
        tool_input: dict,
    ) -> ToolUseBlock:
        raise NotImplementedError

    def build_tool_result(
        self,
        *,
        tool_use_id: str,
        outcome: ToolExecutionOutcome,
    ) -> ToolResultBlock:
        raise NotImplementedError


class ClientToolEventAdapter(ToolEventAdapter):
    def build_tool_use(
        self, *, tool_id: str, tool_name: str, tool_input: dict
    ) -> ToolUseBlock:
        return ClientToolUseBlock(id=tool_id, name=tool_name, input=tool_input)

    def build_tool_result(
        self, *, tool_use_id: str, outcome: ToolExecutionOutcome
    ) -> ToolResultBlock:
        if isinstance(outcome, ToolExecutionFailure):
            return ClientToolResultBlock(
                tool_use_id=tool_use_id,
                content=outcome.error.message,
                is_error=True,
            )
        return ClientToolResultBlock(tool_use_id=tool_use_id, content=outcome.content)


class ServerToolEventAdapter(ToolEventAdapter):
    id_prefix = "srvtoolu"

    def build_tool_use(
        self, *, tool_id: str, tool_name: str, tool_input: dict
    ) -> ToolUseBlock:
        return ServerToolUseBlock(
            id=tool_id,
            name=self.public_tool_name or tool_name,
            input=tool_input,
        )

    def build_tool_result(
        self, *, tool_use_id: str, outcome: ToolExecutionOutcome
    ) -> ToolResultBlock:
        if isinstance(outcome, ToolExecutionFailure):
            return ServerToolResultBlock(
                tool_use_id=tool_use_id,
                content=outcome.error.message,
                is_error=True,
            )
        return ServerToolResultBlock(tool_use_id=tool_use_id, content=outcome.content)


class BashCodeExecutionEventAdapter(ServerToolEventAdapter):
    public_tool_name = "bash_code_execution"

    def build_tool_result(
        self, *, tool_use_id: str, outcome: ToolExecutionOutcome
    ) -> ToolResultBlock:
        if isinstance(outcome, ToolExecutionFailure):
            return BashCodeExecutionToolResultBlock(
                tool_use_id=tool_use_id,
                content=CodeExecutionToolResultErrorBlock(
                    type="bash_code_execution_tool_result_error",
                    error_code="unavailable",
                ),
            )
        return BashCodeExecutionToolResultBlock(
            tool_use_id=tool_use_id,
            content=_single_result(outcome.content, BashCodeExecutionResultBlock),
        )


TextEditorResultTypes = (
    TextEditorCodeExecutionViewResultBlock,
    TextEditorCodeExecutionCreateResultBlock,
    TextEditorCodeExecutionStrReplaceResultBlock,
)
TextEditorResult = (
    TextEditorCodeExecutionViewResultBlock
    | TextEditorCodeExecutionCreateResultBlock
    | TextEditorCodeExecutionStrReplaceResultBlock
)


class TextEditorCodeExecutionEventAdapter(ServerToolEventAdapter):
    public_tool_name = "text_editor_code_execution"

    def build_tool_result(
        self, *, tool_use_id: str, outcome: ToolExecutionOutcome
    ) -> ToolResultBlock:
        if isinstance(outcome, ToolExecutionFailure):
            return TextEditorCodeExecutionToolResultBlock(
                tool_use_id=tool_use_id,
                content=CodeExecutionToolResultErrorBlock(
                    type="text_editor_code_execution_tool_result_error",
                    error_code="unavailable",
                ),
            )
        result = _single_result(outcome.content, TextEditorResultTypes)
        return TextEditorCodeExecutionToolResultBlock(
            tool_use_id=tool_use_id,
            content=cast(TextEditorResult, result),
        )


def _single_result(
    content: list[ResultContentBlockType],
    expected_type: type | tuple[type, ...],
):
    if len(content) != 1 or not isinstance(content[0], expected_type):
        raise ValueError(
            "Specialized tool adapter requires exactly one compatible result block"
        )
    return content[0]
