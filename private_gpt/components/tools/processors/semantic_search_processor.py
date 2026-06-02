from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.tools.builders.semantic_search_builder import (
    SemanticSearchToolBuilder,
)
from private_gpt.components.tools.processors.base import (
    ToolProcessor,
    _get_tool_context,
    _is_unresolved_tool,
    _replace_tool,
    _tool_matches,
)
from private_gpt.components.tools.tool_names import SEMANTIC_SEARCH_TOOL_NAME
from private_gpt.server.utils.artifact_input import IngestedArtifact


@singleton
class SemanticSearchProcessor(ToolProcessor):
    @inject
    def __init__(
        self,
        semantic_search_tool_builder: SemanticSearchToolBuilder,
    ) -> None:
        self._builder = semantic_search_tool_builder

    async def intercept(self, request: ResolvedChatRequest) -> bool:
        for tool in request.tool_config.tools:
            if not _tool_matches(
                tool, SEMANTIC_SEARCH_TOOL_NAME
            ) or not _is_unresolved_tool(tool):
                continue

            tool_context = _get_tool_context(request, tool)
            ingested_artifacts = (
                [a for a in tool_context if isinstance(a, IngestedArtifact)]
                if tool_context
                else None
            )
            if not ingested_artifacts:
                raise ValueError(
                    "Semantic search tool requires an ingested artifact context.",
                )
            if len(ingested_artifacts) > 1:
                raise ValueError("Only one ingested context is supported.")

            resolved = await self._builder.build_tool(
                model_id=request.system.model,
                name=tool.name or SEMANTIC_SEARCH_TOOL_NAME,
                type=tool.type or SEMANTIC_SEARCH_TOOL_NAME + "_v1",
                context_filter=ingested_artifacts[0].context_filter,
                generate_citations=request.citation.enabled,
                validate=request.tool_config.validation_mode,
                token_limit=request.context.maximum_context_length,
            )
            return _replace_tool(request, tool, [resolved])
        return False
