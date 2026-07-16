from typing import TYPE_CHECKING

from injector import inject, singleton

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.ingest_component import IngestComponent
from private_gpt.components.ingest.parse_component import ParseComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.components.sandbox import SandboxComponent
from private_gpt.components.tools.builders.summary_builder import (
    SummarizeWorkflowBuilder,
)
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.tools.builders.database_query_builder import (
        DatabaseQueryToolBuilder,
    )
    from private_gpt.components.tools.builders.semantic_search_builder import (
        SemanticSearchToolBuilder,
    )
    from private_gpt.components.tools.builders.tabular_data_builder import (
        TabularDataToolBuilder,
    )
    from private_gpt.components.tools.builders.web_fetch_builder import (
        WebFetchToolBuilder,
    )
    from private_gpt.components.tools.builders.web_search_builder import (
        WebSearchToolBuilder,
    )
    from private_gpt.components.web.web_scraper_service import WebScraperService
    from private_gpt.components.web.web_search.web_search_service import (
        WebSearchService,
    )


@singleton
class SemanticSearchToolBuilderFactory:
    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        node_store_component: NodeStoreComponent,
        embedding_component: EmbeddingComponent,
        ingest_component: IngestComponent,
        parse_component: ParseComponent,
        prompt_builder_service: PromptBuilderService,
    ) -> None:
        self.settings = settings
        self.llm_component = llm_component
        self.vector_store_component = vector_store_component
        self.node_store_component = node_store_component
        self.embedding_component = embedding_component
        self.ingest_component = ingest_component
        self.parse_component = parse_component
        self.prompt_builder_service = prompt_builder_service

    def create(self) -> "SemanticSearchToolBuilder":
        from private_gpt.components.tools.builders.semantic_search_builder import (
            SemanticSearchToolBuilder,
        )

        return SemanticSearchToolBuilder(
            settings=self.settings,
            llm_component=self.llm_component,
            vector_store_component=self.vector_store_component,
            node_store_component=self.node_store_component,
            embedding_component=self.embedding_component,
            ingest_component=self.ingest_component,
            parse_component=self.parse_component,
            prompt_builder_service=self.prompt_builder_service,
        )


@singleton
class TabularDataToolBuilderFactory:
    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        node_store_component: NodeStoreComponent,
        embedding_component: EmbeddingComponent,
        ingest_component: IngestComponent,
        parse_component: ParseComponent,
        sandbox_component: SandboxComponent,
    ) -> None:
        self.settings = settings
        self.llm_component = llm_component
        self.vector_store_component = vector_store_component
        self.node_store_component = node_store_component
        self.embedding_component = embedding_component
        self.ingest_component = ingest_component
        self.parse_component = parse_component
        self.sandbox_component = sandbox_component

    def create(self) -> "TabularDataToolBuilder":
        from private_gpt.components.tools.builders.tabular_data_builder import (
            TabularDataToolBuilder,
        )

        return TabularDataToolBuilder(
            settings=self.settings,
            llm_component=self.llm_component,
            vector_store_component=self.vector_store_component,
            node_store_component=self.node_store_component,
            embedding_component=self.embedding_component,
            ingest_component=self.ingest_component,
            parse_component=self.parse_component,
            sandbox_component=self.sandbox_component,
        )


@singleton
class DatabaseQueryToolBuilderFactory:
    @inject
    def __init__(self, settings: Settings, llm_component: LLMComponent) -> None:
        self.settings = settings
        self.llm_component = llm_component

    def create(self) -> "DatabaseQueryToolBuilder":
        from private_gpt.components.tools.builders.database_query_builder import (
            DatabaseQueryToolBuilder,
        )

        builder: DatabaseQueryToolBuilder = DatabaseQueryToolBuilder(
            settings=self.settings,
            llm_component=self.llm_component,
        )
        return builder


@singleton
class WebScraperServiceFactory:
    @inject
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create(self) -> "WebScraperService":
        from private_gpt.components.web.web_scraper_service import WebScraperService
        from private_gpt.di import get_global_injector

        # Resolve the singleton so every consumer shares one provider pool.
        return get_global_injector().get(WebScraperService)


@singleton
class WebSearchServiceFactory:
    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        summary_builder: SummarizeWorkflowBuilder,
        web_scraper_service_factory: WebScraperServiceFactory,
    ) -> None:
        self.settings = settings
        self.llm_component = llm_component
        self.summary_builder = summary_builder
        self.web_scraper_service_factory = web_scraper_service_factory

    def create(self) -> "WebSearchService":
        from private_gpt.components.web.web_search.web_search_service import (
            WebSearchService,
        )

        return WebSearchService(
            settings=self.settings,
            scraper_service=self.web_scraper_service_factory.create(),
            llm_component=self.llm_component,
            summary_builder=self.summary_builder,
        )


@singleton
class WebFetchToolBuilderFactory:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        web_scraper_service_factory: WebScraperServiceFactory,
    ) -> None:
        self.llm_component = llm_component
        self.web_scraper_service_factory = web_scraper_service_factory

    def create(self) -> "WebFetchToolBuilder":
        from private_gpt.components.tools.builders.web_fetch_builder import (
            WebFetchToolBuilder,
        )

        return WebFetchToolBuilder(
            llm_component=self.llm_component,
            web_scraper=self.web_scraper_service_factory.create(),
        )


@singleton
class WebSearchToolBuilderFactory:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        web_search_service_factory: WebSearchServiceFactory,
    ) -> None:
        self.llm_component = llm_component
        self.web_search_service_factory = web_search_service_factory

    def create(self) -> "WebSearchToolBuilder":
        from private_gpt.components.tools.builders.web_search_builder import (
            WebSearchToolBuilder,
        )

        return WebSearchToolBuilder(
            llm_component=self.llm_component,
            web_search_service=self.web_search_service_factory.create(),
        )
