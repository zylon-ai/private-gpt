import logging

from injector import inject, singleton
from llama_index.core.base.llms.types import ChatMessage
from pydantic import BaseModel

from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.components.tools.tool_factories import (
    DatabaseQueryToolBuilderFactory,
    SemanticSearchToolBuilderFactory,
    TabularDataToolBuilderFactory,
    WebFetchToolBuilderFactory,
    WebSearchToolBuilderFactory,
)
from private_gpt.events.models import (
    ResultContentBlockType,
    TextBlock,
)
from private_gpt.server.utils.artifact_input import SqlDatabaseArtifact
from private_gpt.settings.settings import Settings, settings

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if settings().server.debug_mode else logging.INFO)


class ToolResponse(BaseModel):
    """Response model for tool operations."""

    content: list[ResultContentBlockType] = []
    is_error: bool = False


@singleton
class ToolService:
    @inject
    def __init__(
        self,
        settings: Settings,
        semantic_search_tool_builder_factory: SemanticSearchToolBuilderFactory,
        tabular_data_tool_builder_factory: TabularDataToolBuilderFactory,
        database_query_tool_builder_factory: DatabaseQueryToolBuilderFactory,
        web_fetch_tool_builder_factory: WebFetchToolBuilderFactory,
        web_search_tool_builder_factory: WebSearchToolBuilderFactory,
    ) -> None:
        self.settings = settings
        self._semantic_search_tool_builder_factory = (
            semantic_search_tool_builder_factory
        )
        self._tabular_data_tool_builder_factory = tabular_data_tool_builder_factory
        self._database_query_tool_builder_factory = database_query_tool_builder_factory
        self._web_fetch_tool_builder_factory = web_fetch_tool_builder_factory
        self._web_search_tool_builder_factory = web_search_tool_builder_factory

    async def semantic_search_tool(
        self,
        query: str,
        context_filter: ContextFilter,
        use_condense: bool = True,
        generate_citations: bool = False,
    ) -> ToolResponse:
        try:
            builder = self._semantic_search_tool_builder_factory.create()

            workflow = await builder.build(
                context_filter=context_filter,
            )

            token_limit = (
                self.settings.chat.maximum_context_length
                if self.settings.chat.maximum_context_length
                else None
            )

            content = await workflow.run_semantic_search(
                query=query,
                use_condense=use_condense,
                generate_citations=generate_citations,
                token_limit=token_limit,
            )
            if not content:
                content = [TextBlock(text="No results found for the query.")]

            return ToolResponse(
                content=content,
                is_error=False,
            )
        except Exception as e:
            logger.error("Error in semantic_search_tool", exc_info=e)
            return ToolResponse(
                content=[TextBlock(text="Error processing semantic_search")],
                is_error=True,
            )

    async def tabular_data_analysis_tool(
        self,
        query: str,
        context_filter: ContextFilter,
        use_condense: bool = True,
        generate_citations: bool = False,
    ) -> ToolResponse:
        try:
            builder = self._tabular_data_tool_builder_factory.create()

            workflow = await builder.build(
                context_filter=context_filter,
            )

            content, is_in_error = await workflow.run_tabular_data_analysis(
                query=query,
                use_condense=use_condense,
                generate_citations=generate_citations,
            )
            return ToolResponse(
                content=content,
                is_error=is_in_error,
            )
        except ImportError as e:
            logger.warning("Tabular tool unavailable: %s", e)
            return ToolResponse(
                content=[TextBlock(text=str(e))],
                is_error=True,
            )
        except Exception as e:
            logger.error("Error in tabular_data_analysis_tool", exc_info=e)
            return ToolResponse(
                content=[TextBlock(text="Error processing tabular data analysis")],
                is_error=True,
            )

    async def database_query_tool(
        self,
        query: str,
        sql_artifacts: list[SqlDatabaseArtifact],
        chat_history: list[ChatMessage] | None = None,
    ) -> ToolResponse:
        try:
            if not sql_artifacts:
                return ToolResponse(
                    content=[TextBlock(text="No SQL database artifacts provided.")],
                    is_error=True,
                )

            builder = self._database_query_tool_builder_factory.create()

            tool = await builder.build_tool(
                sql_artifacts=sql_artifacts,
                chat_history=chat_history,
            )

            response = await tool.async_fn(query)
            return ToolResponse(
                content=response.raw_output,
                is_error=False,
            )
        except Exception as e:
            logger.error("Error in database_query_tool", exc_info=e)
            return ToolResponse(
                content=[TextBlock(text="Error processing database query")],
                is_error=True,
            )

    async def web_search_tool(
        self,
        query: str,
    ) -> ToolResponse:
        try:
            builder = self._web_search_tool_builder_factory.create()

            tool = await builder.build_tool()

            response = await tool.async_fn(query)
            return ToolResponse(
                content=response,
                is_error=False,
            )

        except Exception as e:
            logger.error(f"Error in web_search_tool: {e}")
            return ToolResponse(
                content=[TextBlock(text="Error processing web search")],
                is_error=True,
            )

    async def web_fetch_tool(
        self,
        url: str,
    ) -> ToolResponse:
        try:
            builder = self._web_fetch_tool_builder_factory.create()

            tool = builder.build_tool()

            content = await tool.async_fn(url)
            return ToolResponse(
                content=content,
                is_error=False,
            )
        except Exception as e:
            logger.error("Error in web_fetch_tool", exc_info=e)
            return ToolResponse(
                content=[TextBlock(text="Error processing web fetch")],
                is_error=True,
            )
