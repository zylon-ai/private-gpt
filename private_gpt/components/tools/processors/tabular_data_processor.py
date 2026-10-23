import logging

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ResolvedChatRequest
from private_gpt.components.tools.builders.tabular_data_builder import (
    TabularDataToolBuilder,
)
from private_gpt.components.tools.processors.base import (
    ToolProcessor,
    _get_tool_context,
    _is_unresolved_tool,
    _replace_tool,
    _tool_matches,
)
from private_gpt.components.tools.tool_names import TABULAR_DATA_ANALYSIS
from private_gpt.server.utils.artifact_input import IngestedArtifact

logger = logging.getLogger(__name__)

_PANDASAI_NOT_INSTALLED_MSG = (
    "The tabular data analysis tool is not available because PandasAI is not installed. "
    "Install it with: uv sync --extra tool-tabular"
)


@singleton
class TabularDataProcessor(ToolProcessor):
    @inject
    def __init__(self, tabular_data_tool_builder: TabularDataToolBuilder) -> None:
        self._builder = tabular_data_tool_builder

    async def intercept(self, request: ResolvedChatRequest) -> bool:
        for tool in request.tool_config.tools:
            if not _tool_matches(
                tool, TABULAR_DATA_ANALYSIS
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
                    "Tabular data analysis tool requires an ingested artifact context.",
                )
            if len(ingested_artifacts) > 1:
                raise ValueError("Only one ingested context is supported.")

            try:
                resolved = await self._builder.build_tool(
                    model_id=request.system.model,
                    name=tool.name or TABULAR_DATA_ANALYSIS,
                    type=tool.type or TABULAR_DATA_ANALYSIS + "_v1",
                    context_filter=ingested_artifacts[0].context_filter,
                    validate=request.tool_config.validation_mode,
                    blob_visibility=request.system.blob_visibility,
                )
            except ImportError as e:
                logger.warning("Tabular tool unavailable: %s", e)
                raise RuntimeError(_PANDASAI_NOT_INSTALLED_MSG) from e

            return _replace_tool(request, tool, [resolved])
        return False
