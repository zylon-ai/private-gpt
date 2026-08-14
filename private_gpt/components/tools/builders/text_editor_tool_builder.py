from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import (
    ToolRequirements,
    ToolSpec,
)
from private_gpt.components.code_execution.code_execution_component import (
    CodeExecutionComponent,
)
from private_gpt.components.tools.events.adapters import (
    TextEditorCodeExecutionEventAdapter,
)
from private_gpt.components.tools.remote_execution import build_rebuild_metadata
from private_gpt.components.tools.tool_names import (
    TEXT_EDITOR_CREATE_TOOL_NAME,
    TEXT_EDITOR_INSERT_TOOL_NAME,
    TEXT_EDITOR_STR_REPLACE_TOOL_NAME,
    TEXT_EDITOR_VIEW_TOOL_NAME,
)
from private_gpt.components.tools.tool_placeholders import (
    TEXT_EDITOR_CREATE_TOOL_FN,
    TEXT_EDITOR_INSERT_TOOL_FN,
    TEXT_EDITOR_STR_REPLACE_TOOL_FN,
    TEXT_EDITOR_VIEW_TOOL_FN,
)
from private_gpt.components.tools.utils import truncate_output
from private_gpt.di import get_global_injector
from private_gpt.events.models import (
    NO_TOOL_CONTENT,
    TextEditorCodeExecutionCreateResultBlock,
    TextEditorCodeExecutionStrReplaceResultBlock,
    TextEditorCodeExecutionViewResultBlock,
)
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.code_execution.base import (
        CodeExecutionSession,
        CodeExecutionSessionConfig,
    )
    from private_gpt.events.models import ResultContentBlockType


def _truncated(output: str, max_bytes: int) -> str:
    return truncate_output(output, max_bytes)


@singleton
class TextEditorToolBuilder:
    @inject
    def __init__(
        self,
        code_execution_component: CodeExecutionComponent,
        settings: Settings,
    ) -> None:
        self._component = code_execution_component
        self._settings = settings

    async def _session(
        self, config: CodeExecutionSessionConfig
    ) -> CodeExecutionSession:
        session = await self._component.get_or_create_session(config)
        if session is None:
            raise ValueError("code_execution provider is not configured.")
        return session

    async def build_view_tool(
        self,
        config: CodeExecutionSessionConfig,
        name: str = TEXT_EDITOR_VIEW_TOOL_NAME,
        type: str = TEXT_EDITOR_VIEW_TOOL_NAME + "_v1",
        description: str = TEXT_EDITOR_VIEW_TOOL_FN.metadata.description,
    ) -> ToolSpec:
        async def view(
            path: str,
            view_range: list[int] | None = None,
        ) -> list[ResultContentBlockType]:
            resolved_view_range: tuple[int, int] | None = None
            if view_range is not None:
                if len(view_range) != 2:
                    raise ValueError("view_range must contain exactly two integers")
                resolved_view_range = (view_range[0], view_range[1])

            session = await self._session(config)
            result = await session.view(
                path,
                view_range=resolved_view_range,
            )
            if not result.success:
                raise RuntimeError(result.error or "Unable to view file")
            output = _truncated(
                result.output, self._settings.code_execution.max_output_bytes
            )
            num_lines = len(output.splitlines())
            start_line = resolved_view_range[0] if resolved_view_range else 1
            total_lines = (
                result.total_lines if result.total_lines is not None else num_lines
            )
            return [
                TextEditorCodeExecutionViewResultBlock(
                    content=output if output.strip() else NO_TOOL_CONTENT,
                    num_lines=num_lines,
                    start_line=start_line,
                    total_lines=total_lines,
                )
            ]

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            event_adapter=TextEditorCodeExecutionEventAdapter,
            description=description,
            async_fn=view,
            requirements=[ToolRequirements.SANDBOX],
            execution_metadata=build_rebuild_metadata(
                rebuild_text_editor_view_tool,
                {
                    "config": config,
                    "name": name,
                    "type": type,
                    "description": description,
                },
            ),
        )

    async def build_str_replace_tool(
        self,
        config: CodeExecutionSessionConfig,
        name: str = TEXT_EDITOR_STR_REPLACE_TOOL_NAME,
        type: str = TEXT_EDITOR_STR_REPLACE_TOOL_NAME + "_v1",
        description: str = TEXT_EDITOR_STR_REPLACE_TOOL_FN.metadata.description,
    ) -> ToolSpec:
        async def str_replace(
            path: str,
            old_str: str,
            new_str: str,
        ) -> list[ResultContentBlockType]:
            session = await self._session(config)
            result = await session.str_replace(path, old_str, new_str)
            if not result.success:
                raise RuntimeError(result.error or "Unable to replace text")
            old_lines = old_str.splitlines()
            new_lines = new_str.splitlines()
            diff_lines = [f"- {line}" for line in old_lines] + [
                f"+ {line}" for line in new_lines
            ]
            start = result.start_line or 0
            return [
                TextEditorCodeExecutionStrReplaceResultBlock(
                    old_start=start,
                    new_start=start,
                    old_lines=len(old_lines),
                    new_lines=len(new_lines),
                    lines=diff_lines,
                )
            ]

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            event_adapter=TextEditorCodeExecutionEventAdapter,
            description=description,
            async_fn=str_replace,
            requirements=[ToolRequirements.SANDBOX],
            execution_metadata=build_rebuild_metadata(
                rebuild_text_editor_str_replace_tool,
                {
                    "config": config,
                    "name": name,
                    "type": type,
                    "description": description,
                },
            ),
        )

    async def build_create_tool(
        self,
        config: CodeExecutionSessionConfig,
        name: str = TEXT_EDITOR_CREATE_TOOL_NAME,
        type: str = TEXT_EDITOR_CREATE_TOOL_NAME + "_v1",
        description: str = TEXT_EDITOR_CREATE_TOOL_FN.metadata.description,
    ) -> ToolSpec:
        async def create(
            path: str,
            file_text: str,
        ) -> list[ResultContentBlockType]:
            session = await self._session(config)
            result = await session.create(path, file_text)
            if not result.success:
                raise RuntimeError(result.error or "Unable to create file")
            return [
                TextEditorCodeExecutionCreateResultBlock(
                    is_file_update=result.is_update
                )
            ]

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            event_adapter=TextEditorCodeExecutionEventAdapter,
            description=description,
            async_fn=create,
            requirements=[ToolRequirements.SANDBOX],
            execution_metadata=build_rebuild_metadata(
                rebuild_text_editor_create_tool,
                {
                    "config": config,
                    "name": name,
                    "type": type,
                    "description": description,
                },
            ),
        )

    async def build_insert_tool(
        self,
        config: CodeExecutionSessionConfig,
        name: str = TEXT_EDITOR_INSERT_TOOL_NAME,
        type: str = TEXT_EDITOR_INSERT_TOOL_NAME + "_v1",
        description: str = TEXT_EDITOR_INSERT_TOOL_FN.metadata.description,
    ) -> ToolSpec:
        async def insert(
            path: str,
            insert_line: int,
            new_str: str,
        ) -> list[ResultContentBlockType]:
            session = await self._session(config)
            result = await session.insert(path, insert_line, new_str)
            if not result.success:
                raise RuntimeError(result.error or "Unable to insert text")
            new_lines_list = new_str.splitlines()
            return [
                TextEditorCodeExecutionStrReplaceResultBlock(
                    old_start=insert_line,
                    new_start=insert_line,
                    old_lines=0,
                    new_lines=len(new_lines_list),
                    lines=[f"+ {line}" for line in new_lines_list],
                )
            ]

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            event_adapter=TextEditorCodeExecutionEventAdapter,
            description=description,
            async_fn=insert,
            requirements=[ToolRequirements.SANDBOX],
            execution_metadata=build_rebuild_metadata(
                rebuild_text_editor_insert_tool,
                {
                    "config": config,
                    "name": name,
                    "type": type,
                    "description": description,
                },
            ),
        )


async def rebuild_text_editor_view_tool(**kwargs: Any) -> ToolSpec:
    builder = get_global_injector().get(TextEditorToolBuilder)
    return await builder.build_view_tool(**cast(Any, kwargs))


async def rebuild_text_editor_str_replace_tool(**kwargs: Any) -> ToolSpec:
    builder = get_global_injector().get(TextEditorToolBuilder)
    return await builder.build_str_replace_tool(**cast(Any, kwargs))


async def rebuild_text_editor_create_tool(**kwargs: Any) -> ToolSpec:
    builder = get_global_injector().get(TextEditorToolBuilder)
    return await builder.build_create_tool(**cast(Any, kwargs))


async def rebuild_text_editor_insert_tool(**kwargs: Any) -> ToolSpec:
    builder = get_global_injector().get(TextEditorToolBuilder)
    return await builder.build_insert_tool(**cast(Any, kwargs))
