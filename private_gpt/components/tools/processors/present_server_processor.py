from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.tools.builders.present_server_tool_builder import (
    PresentServerToolBuilder,
)
from private_gpt.components.tools.processors.base import (
    ToolProcessor,
    _is_unresolved_tool,
    _replace_tool,
    _session_id,
    _tool_matches,
)
from private_gpt.components.tools.tool_names import PRESENT_SERVER_TOOL_NAME


@singleton
class PresentServerProcessor(ToolProcessor):
    @inject
    def __init__(self, present_server_tool_builder: PresentServerToolBuilder) -> None:
        self._builder = present_server_tool_builder

    async def intercept(self, request: ResolvedChatRequest) -> bool:
        session_id = _session_id(request)
        for tool in request.tool_config.tools:
            if not _is_unresolved_tool(tool):
                continue
            if _tool_matches(tool, PRESENT_SERVER_TOOL_NAME):
                return _replace_tool(
                    request,
                    tool,
                    [
                        await self._builder.build_tool(
                            session_id,
                            name=tool.name or PRESENT_SERVER_TOOL_NAME,
                            type=tool.type or PRESENT_SERVER_TOOL_NAME + "_v1",
                        )
                    ],
                )
        return False
