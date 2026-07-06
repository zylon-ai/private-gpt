from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.tools.builders.present_files_tool_builder import (
    PresentFilesToolBuilder,
)
from private_gpt.components.tools.processors.base import (
    ToolProcessor,
    _is_unresolved_tool,
    _replace_tool,
    _session_id,
    _tool_matches,
)
from private_gpt.components.tools.tool_names import PRESENT_FILES_TOOL_NAME
from private_gpt.settings.settings import Settings


@singleton
class PresentFilesProcessor(ToolProcessor):
    @inject
    def __init__(
        self,
        present_files_tool_builder: PresentFilesToolBuilder,
        settings: Settings,
    ) -> None:
        self._builder = present_files_tool_builder
        self._enabled = settings.code_execution.tools.present_files_enabled

    async def intercept(self, request: ResolvedChatRequest) -> bool:
        for tool in request.tool_config.tools:
            if not _is_unresolved_tool(tool):
                continue
            if _tool_matches(tool, PRESENT_FILES_TOOL_NAME):
                if not self._enabled:
                    return _replace_tool(request, tool, [])
                session_id = _session_id(request)
                return _replace_tool(
                    request,
                    tool,
                    [
                        await self._builder.build_tool(
                            session_id,
                            bundles=request.context.content_bundles or None,
                            name=tool.name or PRESENT_FILES_TOOL_NAME,
                            type=tool.type or PRESENT_FILES_TOOL_NAME + "_v1",
                        )
                    ],
                )
        return False
