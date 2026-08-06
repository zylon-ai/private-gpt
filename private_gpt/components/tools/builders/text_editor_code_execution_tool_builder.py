from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import (
    ToolRequirements,
    ToolSpec,
)
from private_gpt.components.tools.builders.text_editor_tool_builder import (
    TextEditorToolBuilder,
)
from private_gpt.components.tools.events.adapters import (
    TextEditorCodeExecutionEventAdapter,
)
from private_gpt.components.tools.remote_execution import build_rebuild_metadata
from private_gpt.components.tools.tool_names import (
    TEXT_EDITOR_CODE_EXECUTION_TOOL_NAME,
    TEXT_EDITOR_CREATE_TOOL_NAME,
    TEXT_EDITOR_INSERT_TOOL_NAME,
    TEXT_EDITOR_STR_REPLACE_TOOL_NAME,
    TEXT_EDITOR_VIEW_TOOL_NAME,
)
from private_gpt.components.tools.tool_placeholders import (
    TEXT_EDITOR_CODE_EXECUTION_TOOL_FN,
)
from private_gpt.di import get_global_injector

if TYPE_CHECKING:
    from private_gpt.components.code_execution.base import CodeExecutionSessionConfig
    from private_gpt.events.models import ResultContentBlockType


@singleton
class TextEditorCodeExecutionToolBuilder:
    """Builds the unified text_editor_code_execution tool.

    Delegates to the existing TextEditorToolBuilder children (view, str_replace,
    create, insert) and dispatches at call-time based on the ``command`` parameter.
    No logic is duplicated — each sub-operation is built once and called by name.
    """

    @inject
    def __init__(self, text_editor_tool_builder: TextEditorToolBuilder) -> None:
        self._child_builder = text_editor_tool_builder

    async def build_tool(
        self,
        config: CodeExecutionSessionConfig,
        name: str = TEXT_EDITOR_CODE_EXECUTION_TOOL_NAME,
        type: str = TEXT_EDITOR_CODE_EXECUTION_TOOL_NAME + "_v1",
        description: str = TEXT_EDITOR_CODE_EXECUTION_TOOL_FN.metadata.description,
    ) -> ToolSpec:
        view_tool = await self._child_builder.build_view_tool(config)
        str_replace_tool = await self._child_builder.build_str_replace_tool(config)
        create_tool = await self._child_builder.build_create_tool(config)
        insert_tool = await self._child_builder.build_insert_tool(config)

        _dispatch: dict[str, ToolSpec] = {
            TEXT_EDITOR_VIEW_TOOL_NAME: view_tool,
            TEXT_EDITOR_STR_REPLACE_TOOL_NAME: str_replace_tool,
            TEXT_EDITOR_CREATE_TOOL_NAME: create_tool,
            TEXT_EDITOR_INSERT_TOOL_NAME: insert_tool,
        }

        async def text_editor(
            command: str,
            path: str,
            view_range: list[int] | None = None,
            old_str: str | None = None,
            new_str: str | None = None,
            file_text: str | None = None,
            insert_line: int | None = None,
            insert_text: str | None = None,
        ) -> list[ResultContentBlockType]:
            child = _dispatch.get(command)
            if child is None:
                raise ValueError(
                    f"Unknown text_editor command: {command!r}. "
                    f"Expected one of: {', '.join(_dispatch)}"
                )
            kwargs: dict[str, Any] = {"path": path}
            if command == TEXT_EDITOR_VIEW_TOOL_NAME:
                if view_range is not None:
                    kwargs["view_range"] = view_range
            elif command == TEXT_EDITOR_STR_REPLACE_TOOL_NAME:
                kwargs["old_str"] = old_str
                kwargs["new_str"] = new_str
            elif command == TEXT_EDITOR_CREATE_TOOL_NAME:
                kwargs["file_text"] = file_text
            elif command == TEXT_EDITOR_INSERT_TOOL_NAME:
                kwargs["insert_line"] = insert_line
                kwargs["new_str"] = insert_text if insert_text is not None else new_str
            return await child.async_fn(**kwargs)

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            event_adapter=TextEditorCodeExecutionEventAdapter,
            description=description,
            async_fn=text_editor,
            requirements=[ToolRequirements.SANDBOX],
            execution_metadata=build_rebuild_metadata(
                rebuild_text_editor_code_execution_tool,
                {
                    "config": config,
                    "name": name,
                    "type": type,
                    "description": description,
                },
            ),
        )


async def rebuild_text_editor_code_execution_tool(**kwargs: Any) -> ToolSpec:
    builder = get_global_injector().get(TextEditorCodeExecutionToolBuilder)
    return await builder.build_tool(**cast(Any, kwargs))
