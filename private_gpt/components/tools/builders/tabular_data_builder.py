import asyncio
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Literal

from injector import inject, singleton
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import LLM
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor

from private_gpt.artifact_index.vector_artifact_index import VectorArtifactIndex
from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.chat.input_models import BlobVisibilityMode
from private_gpt.components.chat.models.chat_config_models import (
    ToolRequirements,
    ToolSpec,
)
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.ingest_component import IngestComponent
from private_gpt.components.ingest.parse_component import ParseComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.postprocessor.tree_expansion.table_expansion_post_processor import (
    TableExpansionPostProcessor,
)
from private_gpt.components.sandbox import SandboxComponent
from private_gpt.components.tools.binary_block_decorators import (
    auto_resolve_media_blocks,
)
from private_gpt.components.tools.tool_names import TABULAR_DATA_ANALYSIS
from private_gpt.components.tools.tool_placeholders import TABULAR_DATA_TOOL_FN
from private_gpt.components.tools.types import ToolValidationMode
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.events.models import (
    ResultContentBlockType,
    from_tool_output,
)
from private_gpt.settings.settings import Settings, settings
from private_gpt.utils.dependencies import format_missing_dependency_message

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from private_gpt.components.tabular.pandasai_service import PandasAIService
    from private_gpt.components.workflows.tabular.tabular_data import (
        TabularDataAnalysisInputEvent,
        TabularDataAnalysisWorkflow,
    )

config = settings()


def _load_pandas_ai_service() -> type["PandasAIService"]:
    try:
        from private_gpt.components.tabular.pandasai_service import PandasAIService
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message(
                "Tabular data",
                extras="tool-tabular",
            )
        ) from e

    return PandasAIService


def _load_tabular_workflow_dependencies() -> tuple[
    type["TabularDataAnalysisInputEvent"],
    type["TabularDataAnalysisWorkflow"],
]:
    try:
        from private_gpt.components.workflows.tabular.tabular_data import (
            TabularDataAnalysisInputEvent,
            TabularDataAnalysisWorkflow,
        )
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message(
                "Tabular data",
                extras="tool-tabular",
            )
        ) from e

    return TabularDataAnalysisInputEvent, TabularDataAnalysisWorkflow


@singleton
class TabularDataToolBuilder:
    settings: Settings
    llm_component: LLMComponent
    vector_store_component: VectorStoreComponent
    embedding_component: EmbeddingComponent
    sandbox_component: SandboxComponent

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

    async def _validate_context(
        self, context_filter: ContextFilter | None
    ) -> ContextFilter:
        if not context_filter:
            raise ValueError("context_filter is required")

        if not context_filter.collection:
            raise ValueError("collection is required in context")

        # If artifacts are provided, verify the related required indexes are ready
        # or throw an error
        artifacts = (
            list(set(context_filter.artifacts)) if context_filter.artifacts else None
        )
        if artifacts:
            tasks: list[Coroutine[Any, Any, None]] = []
            for artifact in artifacts:
                vector_artifact_index = VectorArtifactIndex(
                    collection=context_filter.collection,
                    artifact=artifact,
                    vector_store_component=self.vector_store_component,
                    node_store_component=self.node_store_component,
                    embedding_component=self.embedding_component,
                    ingest_component=self.ingest_component,
                    parse_component=self.parse_component,
                )
                tasks.append(vector_artifact_index.apopulated_or_error())
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    raise result

        context_filter = context_filter.model_copy()
        context_filter.artifacts = artifacts

        return context_filter

    def _create_vector_index_retriever(
        self,
        context_filter: ContextFilter,
        embed_model_id: str | None = None,
        top_k: int = config.retrieval.top_k,
    ) -> BaseRetriever:
        collection = context_filter.collection

        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store_component.vector_store(collection),
            index_store=self.node_store_component.index_store(collection),
        )
        index = VectorStoreIndex.from_vector_store(
            self.vector_store_component.vector_store(collection),
            storage_context=storage_context,
            llm=self.llm_component.llm,
            embed_model=self.embedding_component.get_embed(embed_model_id),
            show_progress=False,
        )

        return self.vector_store_component.get_retriever(
            index=index,
            artifacts=context_filter.artifacts,
            collection=context_filter.collection,
            filter_dicts=context_filter.metadata_filter,
            similarity_top_k=top_k,
        )

    def _create_pandas_ai_service(
        self,
    ) -> "PandasAIService":
        """Create a PandasAIService instance."""
        pandas_ai_service_cls = _load_pandas_ai_service()
        return pandas_ai_service_cls(
            llm_component=self.llm_component,
            sandbox_component=self.sandbox_component,
        )

    def _node_postprocessors_fn(
        self,
        context_filter: ContextFilter | None,
        similarity_cutoff: float | None = 0.4,
        **_: Any,
    ) -> Generator[BaseNodePostprocessor, None, None]:

        # Filter nodes by similarity
        if similarity_cutoff:
            yield SimilarityPostprocessor(
                similarity_cutoff=similarity_cutoff,
            )

        # Extend context using tree expansion
        if context_filter:
            yield TableExpansionPostProcessor(
                node_component=self.node_store_component,
                collection=context_filter.collection,
            )

    async def build(
        self,
        context_filter: ContextFilter | None,
        model_id: str | None = None,
        embed_model_id: str | None = None,
        llm: LLM | None = None,
        **kwargs: Any,
    ) -> "TabularDataAnalysisWorkflow":
        _, tabular_data_analysis_workflow_cls = _load_tabular_workflow_dependencies()
        llm = llm or self.llm_component.get_llm(model_id)
        context_filter = await self._validate_context(context_filter)
        retriever = await asyncio.to_thread(
            self._create_vector_index_retriever, context_filter, embed_model_id
        )
        pandas_ai = self._create_pandas_ai_service()

        def node_postprocessors_fn(
            **node_postprocessor_kwargs: Any,
        ) -> list[BaseNodePostprocessor]:
            return list(
                self._node_postprocessors_fn(
                    context_filter=context_filter,
                    llm=llm,
                    **node_postprocessor_kwargs,
                )
            )

        return tabular_data_analysis_workflow_cls(
            llm=llm,
            pandas_ai=pandas_ai,
            retriever=retriever,
            node_postprocessors_fn=node_postprocessors_fn,
            **kwargs,
        )

    async def build_tool(
        self,
        context_filter: ContextFilter | None,
        model_id: str | None = None,
        embed_model_id: str | None = None,
        llm: LLM | None = None,
        name: str = TABULAR_DATA_ANALYSIS,
        type: str = TABULAR_DATA_ANALYSIS + "_v1",
        description: str = TABULAR_DATA_TOOL_FN.metadata.description,
        validate: ToolValidationMode = ToolValidationMode.LAZY,
        runtime: Literal["client", "server"] = "server",
        blob_visibility: BlobVisibilityMode = BlobVisibilityMode.PUBLIC,
        **kwargs: Any,
    ) -> ToolSpec:
        """Builds a tabular search tool."""
        tabular_data_analysis_input_event_cls, _ = _load_tabular_workflow_dependencies()
        lock: asyncio.Lock = asyncio.Lock()
        workflows = {}

        async def _ensure_workflow(
            artifacts: list[str] | None = None,
        ) -> "TabularDataAnalysisWorkflow":
            key: frozenset[str] | None = (
                frozenset(artifacts) if artifacts is not None else None
            )
            if key not in workflows:
                async with lock:
                    if key not in workflows:
                        context_filter_copy = (
                            context_filter.model_copy() if context_filter else None
                        )
                        if artifacts is not None and context_filter_copy is not None:
                            context_filter_copy.artifacts = (
                                list(
                                    set(context_filter_copy.artifacts) & set(artifacts)
                                )
                                if context_filter_copy.artifacts
                                else list(set(artifacts))
                            )
                        elif (
                            context_filter_copy is not None
                            and context_filter_copy.artifacts
                        ):
                            context_filter_copy.artifacts = list(
                                set(context_filter_copy.artifacts)
                            )
                        workflows[key] = await self.build(
                            context_filter=context_filter_copy,
                            model_id=model_id,
                            embed_model_id=embed_model_id,
                            llm=llm,
                        )
            return workflows[key]

        def _sync_format_results(
            content: list[Any],
        ) -> list[ResultContentBlockType]:
            return from_tool_output(content)

        @auto_resolve_media_blocks(blob_visibility=blob_visibility)
        async def tabular_data_analysis(
            query: str,
            artifacts: list[str] | None = None,
        ) -> list[ResultContentBlockType]:

            w = await _ensure_workflow(artifacts=artifacts)
            result = await w.run(
                start_event=tabular_data_analysis_input_event_cls(
                    query=query,
                    kwargs=kwargs,
                )
            )
            return await asyncio.to_thread(
                _sync_format_results,
                result.content,  # type: ignore[attr-defined]
            )

        if validate == ToolValidationMode.EAGER:
            await _ensure_workflow()

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime=runtime,
            description=description,
            async_fn=tabular_data_analysis,
            requirements=[ToolRequirements.SANDBOX],
        )
