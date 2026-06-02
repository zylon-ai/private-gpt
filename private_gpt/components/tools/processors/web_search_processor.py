from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.tools.builders.web_search_builder import (
    WebSearchToolBuilder,
)
from private_gpt.components.tools.processors.base import (
    ToolProcessor,
    _is_unresolved_tool,
    _replace_tool,
    _tool_matches,
)
from private_gpt.components.tools.tool_names import WEB_SEARCH_TOOL_NAME


@singleton
class WebSearchProcessor(ToolProcessor):
    @inject
    def __init__(self, web_search_tool_builder: WebSearchToolBuilder) -> None:
        self._builder = web_search_tool_builder

    async def intercept(self, request: ResolvedChatRequest) -> bool:
        for tool in request.tool_config.tools:
            if not _tool_matches(tool, WEB_SEARCH_TOOL_NAME) or not _is_unresolved_tool(
                tool
            ):
                continue
            resolved = await self._builder.build_tool(
                model_id=request.system.model,
                name=tool.name or WEB_SEARCH_TOOL_NAME,
                type=tool.type or WEB_SEARCH_TOOL_NAME + "_v1",
            )
            return _replace_tool(request, tool, [resolved])
        return False
