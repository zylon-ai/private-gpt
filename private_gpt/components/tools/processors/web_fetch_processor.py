from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.tools.builders.web_fetch_builder import WebFetchToolBuilder
from private_gpt.components.tools.processors.base import (
    ToolProcessor,
    _is_unresolved_tool,
    _replace_tool,
    _tool_matches,
)
from private_gpt.components.tools.tool_names import WEB_FETCH_TOOL_NAME


@singleton
class WebFetchProcessor(ToolProcessor):
    @inject
    def __init__(self, web_fetch_tool_builder: WebFetchToolBuilder) -> None:
        self._builder = web_fetch_tool_builder

    async def intercept(self, request: ResolvedChatRequest) -> bool:
        for tool in request.tool_config.tools:
            if not _tool_matches(tool, WEB_FETCH_TOOL_NAME) or not _is_unresolved_tool(
                tool
            ):
                continue
            resolved = self._builder.build_tool(
                name=tool.name or WEB_FETCH_TOOL_NAME,
                type=tool.type or WEB_FETCH_TOOL_NAME + "_v1",
            )
            return _replace_tool(request, tool, [resolved])
        return False
