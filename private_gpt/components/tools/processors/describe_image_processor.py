from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.code_execution.base import CodeExecutionSessionConfig
from private_gpt.components.tools.builders.describe_image_tool_builder import (
    DescribeImageToolBuilder,
)
from private_gpt.components.tools.processors.base import (
    ToolProcessor,
    _is_unresolved_tool,
    _replace_tool,
    _tool_matches,
    session_id_for,
)
from private_gpt.components.tools.tool_names import DESCRIBE_IMAGE_TOOL_NAME
from private_gpt.server.principal import Principal
from private_gpt.settings.settings import Settings


@singleton
class DescribeImageProcessor(ToolProcessor):
    @inject
    def __init__(
        self,
        describe_image_tool_builder: DescribeImageToolBuilder,
        settings: Settings,
    ) -> None:
        self._builder = describe_image_tool_builder
        self._enabled = settings.code_execution.tools.describe_image.enabled

    async def intercept(self, request: ResolvedChatRequest) -> bool:
        # The placeholder may arrive twice — once from the caller and once from
        # the code_execution fan-out — so resolve the first and drop the rest
        # instead of returning on the first match.
        matches = [
            tool
            for tool in request.tool_config.tools
            if _is_unresolved_tool(tool)
            and _tool_matches(tool, DESCRIBE_IMAGE_TOOL_NAME)
        ]
        if not matches:
            return False

        first, duplicates = matches[0], matches[1:]

        if not self._enabled:
            changed = _replace_tool(request, first, [])
        else:
            config = CodeExecutionSessionConfig(
                session_id=session_id_for(request),
                env=Principal.current().as_env() or {},
                mounts=request.context.mounts or [],
            )
            changed = _replace_tool(
                request,
                first,
                [
                    await self._builder.build_tool(
                        config,
                        name=first.name or DESCRIBE_IMAGE_TOOL_NAME,
                        type=first.type or DESCRIBE_IMAGE_TOOL_NAME + "_v1",
                    )
                ],
            )

        for duplicate in duplicates:
            changed = _replace_tool(request, duplicate, []) or changed

        return changed
