from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.code_execution.base import CodeExecutionSessionConfig
from private_gpt.components.tools.builders.text_editor_tool_builder import (
    TextEditorToolBuilder,
)
from private_gpt.components.tools.processors.base import (
    ToolProcessor,
    _is_unresolved_tool,
    _replace_tool,
    _session_id,
    _tool_matches,
    _wrapper_tool,
)
from private_gpt.components.tools.tool_names import (
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
    def __init__(self, text_editor_tool_builder: TextEditorToolBuilder) -> None:
        self._builder = text_editor_tool_builder

    async def intercept(self, request: ResolvedChatRequest) -> bool:
        self._expand(request)
        return await self._build(request)

    def _expand(self, request: ResolvedChatRequest) -> None:
        expanded = True
        while expanded:
            expanded = False
            for tool in request.tool_config.tools:
                if _is_unresolved_tool(tool) and _tool_matches(
                    tool, TEXT_EDITOR_TOOL_NAME
                ):
                    _replace_tool(
                        request,
                        tool,
                        [
                            _wrapper_tool(TEXT_EDITOR_VIEW_TOOL_NAME),
                            _wrapper_tool(TEXT_EDITOR_STR_REPLACE_TOOL_NAME),
                            _wrapper_tool(TEXT_EDITOR_CREATE_TOOL_NAME),
                            _wrapper_tool(TEXT_EDITOR_INSERT_TOOL_NAME),
                        ],
                    )
                    expanded = True
                    break

    async def _build(self, request: ResolvedChatRequest) -> bool:
        config = CodeExecutionSessionConfig(
            session_id=_session_id(request),
            env=Principal.current().as_env() or {},
            mounts=request.context.mounts or [],
        )

        built_any = False
        for tool in request.tool_config.tools:
            if not _is_unresolved_tool(tool):
                continue
            if _tool_matches(tool, TEXT_EDITOR_VIEW_TOOL_NAME):
                built_any |= _replace_tool(
                    request,
                    tool,
                    [
                        await self._builder.build_view_tool(
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
                        await self._builder.build_str_replace_tool(
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
                        await self._builder.build_create_tool(
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
                        await self._builder.build_insert_tool(
                            config,
                            name=tool.name or TEXT_EDITOR_INSERT_TOOL_NAME,
                            type=tool.type or TEXT_EDITOR_INSERT_TOOL_NAME + "_v1",
                        )
                    ],
                )
        return built_any
