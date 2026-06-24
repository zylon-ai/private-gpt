import logging
from typing import TYPE_CHECKING

from private_gpt.components.web.web_scraper_service import (
    WebScraperService,
)
from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.components.web.web_search.processors.base import (
    BaseWebSearchResultProcessor,
)
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.web.web_scraper_service import (
        WebScraperResult,
    )


logger = logging.getLogger(__name__)


class CleanContentProcessor(BaseWebSearchResultProcessor):
    def __init__(
        self,
        settings: Settings,
        scraper_service: WebScraperService,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._scraper_service = scraper_service

    async def validate(self) -> None:
        if not self._scraper_service.is_initialized:
            raise ValueError(
                "Web fetching is not properly initialized or it is disabled. "
                "Since Web Search depends on web fetching to retrieve content, "
                "the Web Search functionality cannot operate correctly. "
                "Consider enabling web fetching in settings."
            )

    async def process_results(
        self,
        query: str,
        results: list[WebSearchResult],
        model_id: str | None = None,
    ) -> list[WebSearchResult]:
        processed_results: list[WebSearchResult] = []
        for result in results[0 : self._settings.web_search.num_links]:
            try:
                # Get HTML for the result - lazy load scraper service
                response: WebScraperResult = (
                    await self._scraper_service.scrape_max_compress(result.url)
                )
                result.content = response.markdown_content
                response.favicon_url = result.favicon_url
                result.content_type = "text/markdown"
                processed_results.append(result)

            except Exception as e:
                logger.error(f"Error processing {result.url}: {e}", exc_info=True)
                continue

        return processed_results
