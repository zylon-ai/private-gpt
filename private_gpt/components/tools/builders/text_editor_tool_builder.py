from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import (
    ToolRequirements,
    ToolSpec,
)
from private_gpt.components.code_execution.base import CodeExecutionSession
from private_gpt.components.code_execution.code_execution_component import (
    CodeExecutionComponent,
)
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
from private_gpt.events.models import ResultContentBlockType, TextBlock
from private_gpt.settings.settings import Settings


def _truncate_output(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated + "\n...[truncated]"


def _format_output(output: str, max_bytes: int) -> list[ResultContentBlockType]:
    return [TextBlock(text=_truncate_output(output, max_bytes))]


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

    async def _session(self, session_id: str) -> CodeExecutionSession:
        session = await self._component.get_or_create_session(session_id)
        if session is None:
            raise ValueError("code_execution provider is not configured.")
        return session

    async def build_view_tool(
        self,
        session_id: str,
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
                    return _format_output(
                        "Error: view_range must contain exactly two integers.",
                        self._settings.code_execution.max_output_bytes,
                    )
                resolved_view_range = (view_range[0], view_range[1])

            session = await self._session(session_id)
            result = await session.view(
                path,
                view_range=resolved_view_range,
            )
            output = result.output if result.success else f"Error: {result.error}"
            return _format_output(
                output,
                self._settings.code_execution.max_output_bytes,
            )

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            description=description,
            async_fn=view,
            requirements=[ToolRequirements.SANDBOX],
        )

    async def build_str_replace_tool(
        self,
        session_id: str,
        name: str = TEXT_EDITOR_STR_REPLACE_TOOL_NAME,
        type: str = TEXT_EDITOR_STR_REPLACE_TOOL_NAME + "_v1",
        description: str = TEXT_EDITOR_STR_REPLACE_TOOL_FN.metadata.description,
    ) -> ToolSpec:

        async def str_replace(
            path: str,
            old_str: str,
            new_str: str,
        ) -> list[ResultContentBlockType]:
            session = await self._session(session_id)
            result = await session.str_replace(path, old_str, new_str)
            output = result.output if result.success else f"Error: {result.error}"
            return _format_output(
                output,
                self._settings.code_execution.max_output_bytes,
            )

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            description=description,
            async_fn=str_replace,
            requirements=[ToolRequirements.SANDBOX],
        )

    async def build_create_tool(
        self,
        session_id: str,
        name: str = TEXT_EDITOR_CREATE_TOOL_NAME,
        type: str = TEXT_EDITOR_CREATE_TOOL_NAME + "_v1",
        description: str = TEXT_EDITOR_CREATE_TOOL_FN.metadata.description,
    ) -> ToolSpec:

        async def create(
            path: str,
            file_text: str,
        ) -> list[ResultContentBlockType]:
            session = await self._session(session_id)
            result = await session.create(path, file_text)
            output = result.output if result.success else f"Error: {result.error}"
            return _format_output(
                output,
                self._settings.code_execution.max_output_bytes,
            )

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            description=description,
            async_fn=create,
            requirements=[ToolRequirements.SANDBOX],
        )

    async def build_insert_tool(
        self,
        session_id: str,
        name: str = TEXT_EDITOR_INSERT_TOOL_NAME,
        type: str = TEXT_EDITOR_INSERT_TOOL_NAME + "_v1",
        description: str = TEXT_EDITOR_INSERT_TOOL_FN.metadata.description,
    ) -> ToolSpec:

        async def insert(
            path: str,
            insert_line: int,
            new_str: str,
        ) -> list[ResultContentBlockType]:
            session = await self._session(session_id)
            result = await session.insert(path, insert_line, new_str)
            output = result.output if result.success else f"Error: {result.error}"
            return _format_output(
                output,
                self._settings.code_execution.max_output_bytes,
            )

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime="server",
            description=description,
            async_fn=insert,
            requirements=[ToolRequirements.SANDBOX],
        )
