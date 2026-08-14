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
    ErrorDetail,
    ServerToolResultBlock,
    ServerToolUseBlock,
    TextBlock,
    TextEditorCodeExecutionCreateResultBlock,
    TextEditorCodeExecutionStrReplaceResultBlock,
    TextEditorCodeExecutionToolResultBlock,
    TextEditorCodeExecutionViewResultBlock,
    WebFetchResultBlock,
    WebFetchToolResultBlock,
    WebFetchToolResultErrorBlock,
    WebSearchResultBlock,
    WebSearchToolResultBlock,
    WebSearchToolResultError,
    normalize_tool_result_content,
)
from private_gpt.events.models._tool_result_blocks import Renderable

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


def _render_blocks(
    blocks: list[ResultContentBlockType],
) -> list[ResultContentBlockType]:
    return [
        TextBlock(text=block.render()) if isinstance(block, Renderable) else block
        for block in blocks
    ]


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
        content = normalize_tool_result_content(_render_blocks(outcome.content))
        return ClientToolResultBlock(tool_use_id=tool_use_id, content=content)


class ServerToolEventAdapter(ToolEventAdapter):
    id_prefix = "srvtoolu"

    def build_tool_use(
        self, *, tool_id: str, tool_name: str, tool_input: dict
    ) -> ToolUseBlock:
        return ServerToolUseBlock(id=tool_id, name=tool_name, input=tool_input)

    def build_tool_result(
        self, *, tool_use_id: str, outcome: ToolExecutionOutcome
    ) -> ToolResultBlock:
        return self._build_server_result(tool_use_id=tool_use_id, outcome=outcome)

    def _build_server_result(
        self, *, tool_use_id: str, outcome: ToolExecutionOutcome
    ) -> ToolResultBlock:
        if isinstance(outcome, ToolExecutionFailure):
            return ServerToolResultBlock(
                tool_use_id=tool_use_id,
                content=outcome.error.message,
                is_error=True,
            )
        return ServerToolResultBlock(
            tool_use_id=tool_use_id,
            content=normalize_tool_result_content(outcome.content),
        )


class BashCodeExecutionEventAdapter(ServerToolEventAdapter):
    def _build_server_result(
        self, *, tool_use_id: str, outcome: ToolExecutionOutcome
    ) -> ToolResultBlock:
        if isinstance(outcome, ToolExecutionFailure):
            return BashCodeExecutionToolResultBlock(
                tool_use_id=tool_use_id,
                content=CodeExecutionToolResultErrorBlock(
                    type="bash_code_execution_tool_result_error",
                    error_code="unavailable",
                    detail=ErrorDetail(
                        code=outcome.error.code,
                        explanation=outcome.error.message,
                    ),
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
    def _build_server_result(
        self, *, tool_use_id: str, outcome: ToolExecutionOutcome
    ) -> ToolResultBlock:
        if isinstance(outcome, ToolExecutionFailure):
            return TextEditorCodeExecutionToolResultBlock(
                tool_use_id=tool_use_id,
                content=CodeExecutionToolResultErrorBlock(
                    type="text_editor_code_execution_tool_result_error",
                    error_code="unavailable",
                    detail=ErrorDetail(
                        code=outcome.error.code,
                        explanation=outcome.error.message,
                    ),
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
    matches = [block for block in content if isinstance(block, expected_type)]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        "Specialized tool adapter requires exactly one compatible result block"
    )


class WebSearchEventAdapter(ServerToolEventAdapter):
    def _build_server_result(
        self, *, tool_use_id: str, outcome: ToolExecutionOutcome
    ) -> ToolResultBlock:
        if isinstance(outcome, ToolExecutionFailure):
            return WebSearchToolResultBlock(
                tool_use_id=tool_use_id,
                content=WebSearchToolResultError(
                    error_code="unavailable",
                    detail=ErrorDetail(
                        code=outcome.error.code,
                        explanation=outcome.error.message,
                    ),
                ),
            )
        return WebSearchToolResultBlock(
            tool_use_id=tool_use_id,
            content=[
                block
                for block in outcome.content
                if isinstance(block, WebSearchResultBlock)
            ],
        )


class WebFetchEventAdapter(ServerToolEventAdapter):
    def _build_server_result(
        self, *, tool_use_id: str, outcome: ToolExecutionOutcome
    ) -> ToolResultBlock:
        if isinstance(outcome, ToolExecutionFailure):
            return WebFetchToolResultBlock(
                tool_use_id=tool_use_id,
                content=WebFetchToolResultErrorBlock(
                    error_code="unavailable",
                    detail=ErrorDetail(
                        code=outcome.error.code,
                        explanation=outcome.error.message,
                    ),
                ),
            )
        return WebFetchToolResultBlock(
            tool_use_id=tool_use_id,
            content=_single_result(outcome.content, WebFetchResultBlock),
        )


class PresentFilesEventAdapter(ServerToolEventAdapter):
    pass


class PresentServerEventAdapter(ServerToolEventAdapter):
    pass
