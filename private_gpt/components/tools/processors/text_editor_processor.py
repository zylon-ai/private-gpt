from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
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
        session_id = _session_id(request)
        env = Principal.current().as_env() or None
        for tool in request.tool_config.tools:
            if not _is_unresolved_tool(tool):
                continue

            if _tool_matches(tool, TEXT_EDITOR_TOOL_NAME):
                return _replace_tool(
                    request,
                    tool,
                    [
                        _wrapper_tool(TEXT_EDITOR_VIEW_TOOL_NAME),
                        _wrapper_tool(TEXT_EDITOR_STR_REPLACE_TOOL_NAME),
                        _wrapper_tool(TEXT_EDITOR_CREATE_TOOL_NAME),
                        _wrapper_tool(TEXT_EDITOR_INSERT_TOOL_NAME),
                    ],
                )
            bundles = request.context.content_bundles or None
            bundles_to_remove = request.context.bundles_to_remove or None
            if _tool_matches(tool, TEXT_EDITOR_VIEW_TOOL_NAME):
                return _replace_tool(
                    request,
                    tool,
                    [
                        await self._builder.build_view_tool(
                            session_id,
                            bundles=bundles,
                            bundles_to_remove=bundles_to_remove,
                            name=tool.name or TEXT_EDITOR_VIEW_TOOL_NAME,
                            type=tool.type or TEXT_EDITOR_VIEW_TOOL_NAME + "_v1",
                            env=env,
                        )
                    ],
                )
            if _tool_matches(tool, TEXT_EDITOR_STR_REPLACE_TOOL_NAME):
                return _replace_tool(
                    request,
                    tool,
                    [
                        await self._builder.build_str_replace_tool(
                            session_id,
                            bundles=bundles,
                            bundles_to_remove=bundles_to_remove,
                            name=tool.name or TEXT_EDITOR_STR_REPLACE_TOOL_NAME,
                            type=tool.type or TEXT_EDITOR_STR_REPLACE_TOOL_NAME + "_v1",
                            env=env,
                        )
                    ],
                )
            if _tool_matches(tool, TEXT_EDITOR_CREATE_TOOL_NAME):
                return _replace_tool(
                    request,
                    tool,
                    [
                        await self._builder.build_create_tool(
                            session_id,
                            bundles=bundles,
                            bundles_to_remove=bundles_to_remove,
                            name=tool.name or TEXT_EDITOR_CREATE_TOOL_NAME,
                            type=tool.type or TEXT_EDITOR_CREATE_TOOL_NAME + "_v1",
                            env=env,
                        )
                    ],
                )
            if _tool_matches(tool, TEXT_EDITOR_INSERT_TOOL_NAME):
                return _replace_tool(
                    request,
                    tool,
                    [
                        await self._builder.build_insert_tool(
                            session_id,
                            bundles=bundles,
                            bundles_to_remove=bundles_to_remove,
                            name=tool.name or TEXT_EDITOR_INSERT_TOOL_NAME,
                            type=tool.type or TEXT_EDITOR_INSERT_TOOL_NAME + "_v1",
                            env=env,
                        )
                    ],
                )
        return False
