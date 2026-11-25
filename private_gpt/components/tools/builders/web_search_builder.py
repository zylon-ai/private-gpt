from typing import Any, Literal, cast

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.tools.events.adapters import WebSearchEventAdapter
from private_gpt.components.tools.remote_execution import build_rebuild_metadata
from private_gpt.components.tools.tool_names import (
    WEB_SEARCH_TOOL_NAME,
    WEB_SEARCH_TOOL_PROVIDERS,
    WEB_SEARCH_WORKBENCH_LEGACY_TYPE,
)
from private_gpt.components.tools.tool_placeholders import WEB_SEARCH_TOOL_FN
from private_gpt.components.tools.types import ToolValidationMode
from private_gpt.components.web.web_search.web_search_service import WebSearchService
from private_gpt.di import get_global_injector
from private_gpt.events.models import (
    ResultContentBlockType,
    WebSearchResultBlock,
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
        expected_provider = WEB_SEARCH_TOOL_PROVIDERS.get(type)
        legacy_workbench = type == WEB_SEARCH_WORKBENCH_LEGACY_TYPE

        async def validate_search() -> None:
            await self.web_search_service.validate()

        async def run_tool(query: str) -> list[ResultContentBlockType]:
            if legacy_workbench:
                raise ValueError(
                    "Workbench could not identify the web search provider. "
                    "Review the current provider before enabling search again."
                )
            if validate == ToolValidationMode.LAZY:
                # It is not validated because that would imply another call;
                # it is validated directly by making the query.
                pass

            results = await self.web_search_service.search(
                query, model_id=model_id, expected_provider=expected_provider
            )
            return [WebSearchResultBlock.from_web_search_result(r) for r in results]

        if validate == ToolValidationMode.EAGER:
            # At the moment, eager validation is not performed because
            # it would involve a cost (a call would have to be made).
            await validate_search()

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime=runtime,
            event_adapter=WebSearchEventAdapter,
            description=description,
            async_fn=run_tool,
            execution_metadata=build_rebuild_metadata(
                rebuild_workbench_web_search_tool
                if expected_provider is not None or legacy_workbench
                else rebuild_web_search_tool,
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


async def rebuild_workbench_web_search_tool(**kwargs: Any) -> ToolSpec:
    # Older workers cannot rebuild these requests as unrestricted web search.
    return await rebuild_web_search_tool(**kwargs)
