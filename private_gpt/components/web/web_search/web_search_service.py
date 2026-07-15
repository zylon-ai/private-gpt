import logging
from typing import Any

from injector import inject, singleton

from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.tools.builders.summary_builder import (
    SummarizeWorkflowBuilder,
)
from private_gpt.components.web.web_scraper_service import WebScraperService
from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.components.web.web_search.processors.base import (
    BaseWebSearchResultProcessor,
)
from private_gpt.components.web.web_search.processors.select_best_links import (
    SelectBestLinks,
)
from private_gpt.components.web.web_search.providers.base import BaseWebSearchProvider
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class WebSearchService:
    _initialized: bool = False
    _provider: BaseWebSearchProvider
    _processor: BaseWebSearchResultProcessor

    @inject
    def __init__(
        self,
        settings: Settings,
        scraper_service: WebScraperService,
        llm_component: LLMComponent,
        summary_builder: SummarizeWorkflowBuilder,
    ) -> None:
        self._settings = settings
        self._scraper_service = scraper_service
        self._llm_component = llm_component
        self._summary_builder = summary_builder
        self._initialize()

    def _initialize(self) -> None:
        if self._initialized:
            logger.debug("WebSearchService already initialized, skipping")
            return

        if not self._settings.web_search.enabled:
            logger.warning(
                "Web Search is disabled in settings, skipping initialization"
            )
            return

        self._initialize_providers()
        self._initialize_processor()
        self._initialized = True

        logger.debug("WebSearchService initialized successfully")

    async def search(
        self, query: str, model_id: str | None = None, **kwargs: Any
    ) -> list[WebSearchResult]:
        await self.validate()

        provider_results: list[WebSearchResult] = await self._provider.make_query(
            query, self._settings.web_search.num_links, **kwargs
        )
        return await self._processor.process_results(query, provider_results, model_id)

    async def validate(self) -> None:
        if not self._initialized:
            raise ValueError(
                "Web Search is not properly initialized or is disabled in settings."
                "Consider enabling it in the configuration."
            )

        await self._provider.validate()
        await self._processor.validate()

    async def close(self) -> None:
        await self._provider.close()

    def _initialize_providers(self) -> None:
        if self._settings.web_search.provider == "brave":
            from private_gpt.components.web.web_search.providers.brave import (
                BraveSearchProvider,
            )

            self._provider = BraveSearchProvider(self._settings)
        elif self._settings.web_search.provider == "mock":
            from private_gpt.components.web.web_search.providers.mock import (
                MockSearchProvider,
            )

            self._provider = MockSearchProvider()
        else:
            logger.error(
                f"Unsupported web search provider: {self._settings.web_search.provider}"
            )
            raise ValueError(
                f"Unsupported web search provider: {self._settings.web_search.provider}"
            )

        if self._settings.web_search.cached:
            from private_gpt.components.web.web_search.providers.cached import (
                CachedProvider,
            )

            self._provider = CachedProvider(self._provider)

    def _initialize_processor(self) -> None:
        if self._settings.web_search.processor == "simple_text":
            from private_gpt.components.web.web_search.processors.simple_text import (
                SimpleTextProcessor,
            )

            self._processor = SimpleTextProcessor(self._settings)

        elif self._settings.web_search.processor == "scraped_content":
            from private_gpt.components.web.web_search.processors.scraped_content_processor import (
                ScrapedContentProcessor,
            )

            self._processor = ScrapedContentProcessor(settings=self._settings)

        elif self._settings.web_search.processor == "clean_content":
            from private_gpt.components.web.web_search.processors.clean_content import (
                CleanContentProcessor,
            )

            self._processor = CleanContentProcessor(
                settings=self._settings,
                scraper_service=self._scraper_service,
            )
        elif self._settings.web_search.processor == "best_links":
            from private_gpt.components.web.web_search.processors.clean_content import (
                CleanContentProcessor,
            )

            self._processor = SelectBestLinks(
                settings=self._settings,
                scraper_service=self._scraper_service,
                llm_component=self._llm_component,
                summary_builder=self._summary_builder,
            )

        else:
            raise ValueError(
                f"Unknown processor: {self._settings.web_search.processor}. "
                f"Available: simple_text, scraped_content, clean_content"
            )
