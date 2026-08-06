from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.code_execution.base import CodeExecutionSessionConfig
from private_gpt.components.tools.builders.text_editor_code_execution_tool_builder import (
    TextEditorCodeExecutionToolBuilder,
)
from private_gpt.components.tools.builders.text_editor_tool_builder import (
    TextEditorToolBuilder,
)
from private_gpt.components.tools.processors.base import (
    ToolProcessor,
    _is_unresolved_tool,
    _replace_tool,
    _session_id,
    _tool_matches,
)
from private_gpt.components.tools.tool_names import (
    TEXT_EDITOR_CODE_EXECUTION_TOOL_NAME,
    TEXT_EDITOR_CREATE_TOOL_NAME,
    TEXT_EDITOR_INSERT_TOOL_NAME,
    TEXT_EDITOR_STR_REPLACE_TOOL_NAME,
    TEXT_EDITOR_TOOL_NAME,
    TEXT_EDITOR_VIEW_TOOL_NAME,
)
from private_gpt.server.principal import Principal


@singleton
class TextEditorProcessor(ToolProcessor):
    @inject
    def __init__(
        self,
        text_editor_tool_builder: TextEditorToolBuilder,
        text_editor_code_execution_tool_builder: TextEditorCodeExecutionToolBuilder,
    ) -> None:
        self._child_builder = text_editor_tool_builder
        self._unified_builder = text_editor_code_execution_tool_builder

    async def intercept(self, request: ResolvedChatRequest) -> bool:
        return await self._build(request)

    async def _build(self, request: ResolvedChatRequest) -> bool:
        config = CodeExecutionSessionConfig(
            session_id=_session_id(request),
            extra_bundles=request.context.content_bundles or [],
            bundles_to_remove=request.context.bundles_to_remove or [],
            env=Principal.current().as_env() or {},
        )

        built_any = False
        for tool in request.tool_config.tools:
            if not _is_unresolved_tool(tool):
                continue

            if _tool_matches(
                tool, TEXT_EDITOR_TOOL_NAME, TEXT_EDITOR_CODE_EXECUTION_TOOL_NAME
            ):
                built_any |= _replace_tool(
                    request,
                    tool,
                    [
                        await self._unified_builder.build_tool(
                            config,
                            name=tool.name or TEXT_EDITOR_CODE_EXECUTION_TOOL_NAME,
                            type=tool.type
                            or TEXT_EDITOR_CODE_EXECUTION_TOOL_NAME + "_v1",
                        )
                    ],
                )
            elif _tool_matches(tool, TEXT_EDITOR_VIEW_TOOL_NAME):
                built_any |= _replace_tool(
                    request,
                    tool,
                    [
                        await self._child_builder.build_view_tool(
                            config,
                            name=tool.name or TEXT_EDITOR_VIEW_TOOL_NAME,
                            type=tool.type or TEXT_EDITOR_VIEW_TOOL_NAME + "_v1",
                        )
                    ],
                )
            elif _tool_matches(tool, TEXT_EDITOR_STR_REPLACE_TOOL_NAME):
                built_any |= _replace_tool(
                    request,
                    tool,
                    [
                        await self._child_builder.build_str_replace_tool(
                            config,
                            name=tool.name or TEXT_EDITOR_STR_REPLACE_TOOL_NAME,
                            type=tool.type or TEXT_EDITOR_STR_REPLACE_TOOL_NAME + "_v1",
                        )
                    ],
                )
            elif _tool_matches(tool, TEXT_EDITOR_CREATE_TOOL_NAME):
                built_any |= _replace_tool(
                    request,
                    tool,
                    [
                        await self._child_builder.build_create_tool(
                            config,
                            name=tool.name or TEXT_EDITOR_CREATE_TOOL_NAME,
                            type=tool.type or TEXT_EDITOR_CREATE_TOOL_NAME + "_v1",
                        )
                    ],
                )
            elif _tool_matches(tool, TEXT_EDITOR_INSERT_TOOL_NAME):
                built_any |= _replace_tool(
                    request,
                    tool,
                    [
                        await self._child_builder.build_insert_tool(
                            config,
                            name=tool.name or TEXT_EDITOR_INSERT_TOOL_NAME,
                            type=tool.type or TEXT_EDITOR_INSERT_TOOL_NAME + "_v1",
                        )
                    ],
                )
        return built_any
