import asyncio
from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Literal

from injector import inject, singleton
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import LLM
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

from private_gpt.artifact_index.vector_artifact_index import VectorArtifactIndex
from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.engines.citations.types import Document
from private_gpt.components.ingest.ingest_component import IngestComponent
from private_gpt.components.ingest.parse_component import ParseComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.postprocessor.tree_expansion.tree_expansion_replacement_post_processor import (
    TreeExpansionReplacementPostProcessor,
)
from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.components.tools.tool_names import SEMANTIC_SEARCH_TOOL_NAME
from private_gpt.components.tools.tool_placeholders import SEMANTIC_SEARCH_TOOL_FN
from private_gpt.components.tools.types import ToolValidationMode
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.components.workflows.retrieval.semantic_search import (
    SemanticSearchInputEvent,
    SemanticSearchWorkflow,
)
from private_gpt.events.models import (
    ResultContentBlockType,
    SourceBlock,
    TextBlock,
    from_tool_output,
)
from private_gpt.settings.settings import Settings, settings
from private_gpt.utils.token import calculate_maximum_token_expansion

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from private_gpt.components.workflows.retrieval.semantic_search import (
        SemanticSearchResultEvent,
    )

config = settings()


@singleton
class SemanticSearchToolBuilder:
    settings: Settings
    llm_component: LLMComponent
    vector_store_component: VectorStoreComponent
    embedding_component: EmbeddingComponent
    prompt_builder_service: PromptBuilderService

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
            collection=collection,
            filter_dicts=context_filter.metadata_filter,
            similarity_top_k=top_k,
        )

    def _node_postprocessors_fn(
        self,
        llm: LLM,
        tokenizer: TokenizerBase | None,
        context_filter: ContextFilter | None,
        similarity_cutoff: float | None = 0.4,
        token_limit: int | None = None,
        **_: Any,
    ) -> Generator[BaseNodePostprocessor, None, None]:

        # Filter nodes by similarity
        if similarity_cutoff:
            yield SimilarityPostprocessor(
                similarity_cutoff=similarity_cutoff,
            )

        # Extend context using tree expansion
        if context_filter:
            token_limit = calculate_maximum_token_expansion(
                token_limit=token_limit,
                context_window=llm.metadata.context_window,
                maximum_context_length=self.settings.chat.maximum_context_length,
            )
            yield TreeExpansionReplacementPostProcessor(
                node_component=self.node_store_component,
                collection=context_filter.collection,
                token_limit=token_limit,
                tokenizer_fn=tokenizer,
            )

    async def build(
        self,
        context_filter: ContextFilter | None,
        model_id: str | None = None,
        embed_model_id: str | None = None,
        llm: LLM | None = None,
        tokenizer: TokenizerBase | None = None,
    ) -> SemanticSearchWorkflow:
        context_filter = await self._validate_context(context_filter)
        retriever = await asyncio.to_thread(
            self._create_vector_index_retriever, context_filter, embed_model_id
        )

        if not llm:
            llm = self.llm_component.get_llm(model_id)
            tokenizer = self.llm_component.get_tokenizer(model_id)

        def node_postprocessors_fn(
            **node_postprocessor_kwargs: Any,
        ) -> list[BaseNodePostprocessor]:
            return list(
                self._node_postprocessors_fn(
                    llm=llm,
                    tokenizer=tokenizer,
                    context_filter=context_filter,
                    **node_postprocessor_kwargs,
                )
            )

        return SemanticSearchWorkflow(
            llm=llm,
            retriever=retriever,
            node_postprocessors_fn=node_postprocessors_fn,
            prompt_builder_service=self.prompt_builder_service,
        )

    async def build_tool(
        self,
        context_filter: ContextFilter | None,
        model_id: str | None = None,
        embed_model_id: str | None = None,
        name: str = SEMANTIC_SEARCH_TOOL_NAME,
        type: str = SEMANTIC_SEARCH_TOOL_NAME + "_v1",
        description: str = SEMANTIC_SEARCH_TOOL_FN.metadata.description,
        validate: ToolValidationMode = ToolValidationMode.LAZY,
        runtime: Literal["client", "server"] = "server",
        **kwargs: Any,
    ) -> ToolSpec:
        """Builds a semantic search tool."""
        lock: asyncio.Lock = asyncio.Lock()
        workflows: dict[frozenset[str] | None, SemanticSearchWorkflow] = {}

        llm = self.llm_component.get_llm(model_id)
        tokenizer = self.llm_component.get_tokenizer(model_id)

        generate_citations = kwargs.get("generate_citations", False)
        token_limit = kwargs.pop("token_limit") if "token_limit" in kwargs else None

        async def _ensure_workflow(
            artifacts: list[str] | None = None,
        ) -> SemanticSearchWorkflow:
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
                            embed_model_id=embed_model_id,
                            llm=llm,
                            tokenizer=tokenizer,
                        )
            return workflows[key]

        def _sync_format_result(
            nodes: list[NodeWithScore] | None = None,
        ) -> list[ResultContentBlockType]:
            content_blocks = from_tool_output(nodes)
            documents = [
                Document.from_source(source)
                for content_block in content_blocks
                if isinstance(content_block, SourceBlock)
                for source in content_block.sources
            ]
            prompt, _ = self.prompt_builder_service.create_context_prompt(
                documents=documents,
                generate_citations=generate_citations,
                token_limit=token_limit,
                tokenizer_fn=tokenizer,
            )
            return [
                *content_blocks,
                TextBlock(text=prompt.format() or "No content is available."),
            ]

        async def semantic_search(
            query: str,
            artifacts: list[str] | None = None,
        ) -> list[ResultContentBlockType]:

            w = await _ensure_workflow(artifacts=artifacts)
            result: SemanticSearchResultEvent = await w.run(
                start_event=SemanticSearchInputEvent(
                    query=query,
                    use_condense=True,
                    token_limit=token_limit,
                    kwargs=kwargs,
                )
            )
            return await asyncio.to_thread(
                _sync_format_result,
                result.retrieval.nodes,
            )

        async def format_result(
            content_blocks: list[ResultContentBlockType],
        ) -> str:
            text_block = next(
                (block for block in content_blocks if isinstance(block, TextBlock)),
                None,
            )
            return text_block.text if text_block else "No content is available."

        if validate == ToolValidationMode.EAGER:
            await _ensure_workflow()

        return ToolSpec.from_defaults(
            name=name,
            type=type,
            runtime=runtime,
            description=description,
            async_fn=semantic_search,
            async_callback=format_result,
        )
