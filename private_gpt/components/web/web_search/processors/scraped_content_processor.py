import asyncio
import logging

from private_gpt.components.web.web_scraper_service import WebScraperService
from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.components.web.web_search.processors.base import (
    BaseWebSearchResultProcessor,
)
from private_gpt.di import get_global_injector
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


class ScrapedContentProcessor(BaseWebSearchResultProcessor):
    _scraper_service: WebScraperService | None

    def __init__(
        self,
        settings: Settings,
    ):
        super().__init__()
        self._settings = settings
        self._scraper_service = None
        self._initialize()

    def _initialize(self) -> None:
        if self._scraper_service is None:
            self._scraper_service = get_global_injector().get(WebScraperService)

    async def process_results(
        self,
        query: str,
        results: list[WebSearchResult],
        model_id: str | None = None,
    ) -> list[WebSearchResult]:
        if self._scraper_service is None:
            raise RuntimeError("Scraper service not initialized")

        limited_results = results[0 : self._settings.web_search.num_links]

        tasks = [self._scraper_service.scrape(result.url) for result in limited_results]
        scraped_contents = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, (result, scraped_content) in enumerate(
            zip(limited_results, scraped_contents, strict=False), 1
        ):
            result.idx = idx

            if isinstance(scraped_content, Exception):
                logger.warning(
                    f"ScrapedContentProcessor: Failed to scrape {result.url}: {scraped_content}"
                )
                result.content = f"Failed to scrape content: {scraped_content!s}"
                result.is_in_error = True
            else:
                logger.debug(
                    f"ScrapedContentProcessor: Successfully scraped {result.url}"
                )
                result.content = str(scraped_content)
                result.content_type = "text/markdown"

        return limited_results
