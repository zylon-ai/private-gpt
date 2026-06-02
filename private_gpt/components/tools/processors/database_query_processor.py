from injector import inject, singleton
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.tools.builders.database_query_builder import (
    DatabaseQueryToolBuilder,
)
from private_gpt.components.tools.processors.base import (
    ToolProcessor,
    _get_tool_context,
    _is_unresolved_tool,
    _replace_tool,
    _tool_matches,
)
from private_gpt.components.tools.tool_names import DATABASE_QUERY_TOOL_NAME
from private_gpt.server.utils.artifact_input import SqlDatabaseArtifact


@singleton
class DatabaseQueryProcessor(ToolProcessor):
    @inject
    def __init__(
        self,
        database_query_tool_builder: DatabaseQueryToolBuilder,
    ) -> None:
        self._builder = database_query_tool_builder

    async def intercept(self, request: ResolvedChatRequest) -> bool:
        for tool in request.tool_config.tools:
            if not _tool_matches(
                tool, DATABASE_QUERY_TOOL_NAME
            ) or not _is_unresolved_tool(tool):
                continue

            tool_context = _get_tool_context(request, tool)
            sql_artifacts = [
                ctx for ctx in tool_context if isinstance(ctx, SqlDatabaseArtifact)
            ]
            if not sql_artifacts:
                raise ValueError(
                    "Database query tool requires at least one SQL database artifact in the tool context.",
                )

            chat_history = request.messages.copy()
            prompt_blocks = request.system.get_prompt()
            if prompt_blocks:
                chat_history.insert(
                    0,
                    ChatMessage(role=MessageRole.SYSTEM, blocks=prompt_blocks),
                )

            resolved = await self._builder.build_tool(
                name=tool.name or DATABASE_QUERY_TOOL_NAME,
                type=tool.type or DATABASE_QUERY_TOOL_NAME + "_v1",
                sql_artifacts=sql_artifacts,
                chat_history=chat_history,
                validate=request.tool_config.validation_mode,
                blob_visibility=request.system.blob_visibility,
            )
            return _replace_tool(request, tool, [resolved])
        return False
