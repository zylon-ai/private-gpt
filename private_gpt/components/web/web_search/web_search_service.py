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
from private_gpt.observability.tracing import (
    SpanKind,
    retrieval_documents,
    set_span_attributes,
    start_span,
)
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
        self,
        query: str,
        model_id: str | None = None,
        *,
        expected_provider: str | None = None,
        expected_automatic_provider: bool | None = None,
        **kwargs: Any,
    ) -> list[WebSearchResult]:
        if (
            expected_provider is not None
            and self._settings.web_search.provider != expected_provider
        ):
            raise ValueError(
                "The backend's web search provider differs from the expected provider. "
                "Review the current provider before enabling search again."
            )
        await self.validate()
        if (
            expected_automatic_provider is not None
            and self._settings.web_search.automatic_provider
            != expected_automatic_provider
        ):
            if expected_provider is None:
                raise ValueError(
                    "Select the search provider in the chat tool type or configure it "
                    "explicitly on the backend before running web search."
                )
            raise ValueError(
                "The backend's web search provider selection changed. "
                "Review the current provider before enabling search again."
            )

        provider = type(self._provider).__name__
        processor = type(self._processor).__name__
        with start_span(
            f"web_search.query {provider}",
            SpanKind.RETRIEVER,
            input=query,
            attributes={"pgpt.web_search.provider": provider},
        ) as span:
            provider_results: list[WebSearchResult] = await self._provider.make_query(
                query, self._settings.web_search.num_links, **kwargs
            )
            set_span_attributes(
                span,
                **{"pgpt.web_search.results": len(provider_results)},
                **retrieval_documents(_web_documents(provider_results)),
            )
        with start_span(
            f"web_search.process {processor}",
            SpanKind.CHAIN,
            input={"query": query, "urls": [r.url for r in provider_results]},
            attributes={"pgpt.web_search.processor": processor},
        ) as span:
            processed = await self._processor.process_results(
                query, provider_results, model_id
            )
            set_span_attributes(
                span,
                output=[
                    r.model_dump(exclude_none=True, exclude={"metadata"})
                    for r in processed
                ],
                **{"pgpt.web_search.results": len(processed)},
            )
        return processed

    async def validate(self) -> None:
        if not self._initialized:
            raise ValueError(
                "Web Search is not properly initialized or is disabled in settings."
                "Consider enabling it in the configuration."
            )

        self._validate_offline_mode(self._settings)
        await self._provider.validate()
        await self._processor.validate()

    @classmethod
    async def validate_configuration(cls, settings: Settings) -> None:
        """Check availability without initializing the model or making a query."""
        if not settings.web_search.enabled:
            raise ValueError("Web search is disabled in backend settings.")
        cls._validate_offline_mode(settings)
        await cls._create_provider(settings).validate()
        if settings.web_search.processor in {"best_links", "clean_content"} and (
            not settings.web_fetch.enabled
        ):
            raise ValueError("The configured search processor requires URL fetching.")

    @staticmethod
    def _validate_offline_mode(settings: Settings) -> None:
        web_search = settings.web_search
        if settings.server.network.offline_mode and (
            web_search.provider != "mock" or web_search.processor != "simple_text"
        ):
            raise ValueError("Live web search is unavailable in offline mode.")

    async def close(self) -> None:
        await self._provider.close()

    @staticmethod
    def _create_provider(settings: Settings) -> BaseWebSearchProvider:
        provider = settings.web_search.provider
        if provider == "brave":
            from private_gpt.components.web.web_search.providers.brave import (
                BraveSearchProvider,
            )

            return BraveSearchProvider(settings)
        if provider == "mock":
            from private_gpt.components.web.web_search.providers.mock import (
                MockSearchProvider,
            )

            return MockSearchProvider()
        if provider == "parallel":
            from private_gpt.components.web.web_search.providers.parallel import (
                ParallelSearchProvider,
            )

            return ParallelSearchProvider(settings)
        raise ValueError(f"Unsupported web search provider: {provider}")

    def _initialize_providers(self) -> None:
        self._provider = self._create_provider(self._settings)
        if self._settings.web_search.cached:
            from private_gpt.components.web.web_search.providers.cached import (
                CachedProvider,
            )

            if self._settings.web_search.provider == "parallel":
                self._provider = CachedProvider(
                    self._provider, cache_dir=".cache/web_search/parallel"
                )
            else:
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


def _web_documents(results: list[WebSearchResult]) -> list[dict[str, Any]]:
    return [
        {
            "id": result.url,
            "content": result.content or result.description,
            "metadata": {"title": result.title, "url": result.url, "age": result.age},
        }
        for result in results
    ]
