import asyncio
from typing import Any, Literal, cast

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.chunk.models import Website
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.tools.remote_execution import build_rebuild_metadata
from private_gpt.components.tools.tool_names import WEB_SEARCH_TOOL_NAME
from private_gpt.components.tools.tool_placeholders import WEB_SEARCH_TOOL_FN
from private_gpt.components.tools.types import ToolValidationMode
from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.components.web.web_search.web_search_service import WebSearchService
from private_gpt.di import get_global_injector
from private_gpt.events.models import (
    ResultContentBlockType,
    TextBlock,
    from_tool_output,
)


@singleton
class WebSearchToolBuilder:
    """A builder class for creating a web search tool.

    This tools allows users to search the web for a given query.
    It retrieves search results and processes them to extract relevant information.
    """

    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        web_search_service: WebSearchService,
    ):
        """Initialize the WebSearchToolBuilder with necessary components."""
        self.llm_component = llm_component
        self.web_search_service = web_search_service

    async def build_tool(
        self,
        model_id: str | None = None,
        name: str = WEB_SEARCH_TOOL_NAME,
        type: str = WEB_SEARCH_TOOL_NAME + "_v1",
        description: str = WEB_SEARCH_TOOL_FN.metadata.description,
        validate: ToolValidationMode = ToolValidationMode.LAZY,
        runtime: Literal["client", "server"] = "server",
    ) -> ToolSpec:
        async def validate_search() -> None:
            await self.web_search_service.validate()

        def _sync_format_results(
            content: list[WebSearchResult],
        ) -> list[ResultContentBlockType]:
            if not content:
                return [
                    TextBlock(
                        text="No results found for the given query.",
                    )
                ]

            websites = [Website.from_website_result(res) for res in content]
            return [
                *from_tool_output(websites),
                *[TextBlock(text=str(result)) for result in content],
            ]

        async def run_tool(query: str) -> list[ResultContentBlockType]:
            if validate == ToolValidationMode.LAZY:
                # It is not validated because that would imply another call;
                # it is validated directly by making the query.
                pass

            results = await self.web_search_service.search(query, model_id=model_id)
            return await asyncio.to_thread(
                _sync_format_results,
                results,
            )

        if validate == ToolValidationMode.EAGER:
            # At the moment, eager validation is not performed because
            # it would involve a cost (a call would have to be made).
            await validate_search()

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime=runtime,
            description=description,
            async_fn=run_tool,
            execution_metadata=build_rebuild_metadata(
                rebuild_web_search_tool,
                {
                    "model_id": model_id,
                    "name": name,
                    "type": type,
                    "description": description,
                    "validate": validate,
                    "runtime": runtime,
                },
            ),
        )


async def rebuild_web_search_tool(**kwargs: Any) -> ToolSpec:
    builder = get_global_injector().get(WebSearchToolBuilder)
    return await builder.build_tool(**cast(Any, kwargs))
