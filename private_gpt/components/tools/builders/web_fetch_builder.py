from typing import Literal

from injector import inject, singleton

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.tools.tool_names import WEB_FETCH_TOOL_NAME
from private_gpt.components.tools.tool_placeholders import WEB_FETCH_TOOL_FN
from private_gpt.components.web.web_scraper_service import WebScraperService
from private_gpt.events.models import (
    ResultContentBlockType,
    TextBlock,
)


@singleton
class WebFetchToolBuilder:
    """A builder class for creating a web fetch tool.

    This tool allows users to fetch and summarize content from a given URL.
    It retrieves the content of the webpage
    and processes it to extract relevant information.
    """

    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        web_scraper: WebScraperService,
    ):
        """Initialize the WebFetchToolBuilder with necessary components."""
        self.llm_component = llm_component
        self.web_scraper = web_scraper

    def build_tool(
        self,
        name: str = WEB_FETCH_TOOL_NAME,
        type: str = WEB_FETCH_TOOL_NAME + "_v1",
        description: str = WEB_FETCH_TOOL_FN.metadata.description,
        runtime: Literal["client", "server"] = "server",
    ) -> ToolSpec:
        async def run_tool(url: str) -> list[ResultContentBlockType]:

            result = await self.web_scraper.scrape_max_compress(url)

            if not result.markdown_content:
                return [
                    TextBlock(
                        text="No content could be fetched from the provided URL.",
                    )
                ]

            page_content = result.markdown_content
            return [
                TextBlock(
                    text=page_content,
                )
            ]

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime=runtime,
            description=description,
            async_fn=run_tool,
        )
