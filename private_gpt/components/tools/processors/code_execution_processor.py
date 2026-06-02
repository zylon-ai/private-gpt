from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.tools.processors.base import (
    ToolProcessor,
    _is_unresolved_tool,
    _replace_tool,
    _tool_matches,
    _wrapper_tool,
)
from private_gpt.components.tools.tool_names import (
    BASH_TOOL_NAME,
    CODE_EXECUTION_TOOL_NAME,
    TEXT_EDITOR_TOOL_NAME,
)


@singleton
class CodeExecutionProcessor(ToolProcessor):
    @inject
    def __init__(self) -> None:
        pass

    async def intercept(self, request: ResolvedChatRequest) -> bool:
        for tool in request.tool_config.tools:
            if not _tool_matches(
                tool, CODE_EXECUTION_TOOL_NAME
            ) or not _is_unresolved_tool(tool):
                continue

            return _replace_tool(
                request,
                tool,
                [
                    _wrapper_tool(
                        BASH_TOOL_NAME,
                    ),
                    _wrapper_tool(
                        TEXT_EDITOR_TOOL_NAME,
                    ),
                ],
            )
        return False
