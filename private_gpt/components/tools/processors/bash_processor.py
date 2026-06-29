from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.tools.builders.bash_tool_builder import BashToolBuilder
from private_gpt.components.tools.processors.base import (
    ToolProcessor,
    _is_unresolved_tool,
    _replace_tool,
    _session_id,
    _tool_matches,
)
from private_gpt.components.tools.tool_names import BASH_TOOL_NAME


@singleton
class BashProcessor(ToolProcessor):
    @inject
    def __init__(self, bash_tool_builder: BashToolBuilder) -> None:
        self._bash_builder = bash_tool_builder

    async def intercept(self, request: ResolvedChatRequest) -> bool:
        for tool in request.tool_config.tools:
            if not _tool_matches(tool, BASH_TOOL_NAME) or not _is_unresolved_tool(tool):
                continue

            resolved = await self._bash_builder.build_tool(
                _session_id(request),
                name=tool.name or BASH_TOOL_NAME,
                type=tool.type or BASH_TOOL_NAME + "_v1",
                bundles=request.context.content_bundles or None,
            )
            return _replace_tool(request, tool, [resolved])
        return False
