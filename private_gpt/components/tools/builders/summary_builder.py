import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

from injector import inject, singleton
from llama_index.core.llms import LLM
from pydantic import BaseModel

from private_gpt.artifact_index.vector_artifact_index import VectorArtifactIndex
from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.ingest_component import IngestComponent
from private_gpt.components.ingest.parse_component import ParseComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.components.tools.tool_names import SUMMARIZE_TOOL_NAME
from private_gpt.components.tools.tool_placeholders import SUMMARIZE_TOOL_FN
from private_gpt.components.tools.types import ToolValidationMode
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.components.workflows.others.summary import (
    SummarizeInputEvent,
    SummarizeWorkflow,
)
from private_gpt.components.workflows.others.summary_retriever import (
    CompositeRetriever,
    ContextRetriever,
    InMemoryRetriever,
)
from private_gpt.events.models import ResultContentBlockType, from_tool_output
from private_gpt.server.content.content_service import ContentService
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.components.workflows.others.summary import (
        SummarizeResultEvent,
    )
    from private_gpt.components.workflows.others.summary_retriever import (
        Retriever,
    )


@singleton
class SummarizeWorkflowBuilder:
    """Builder for creating summary workflows and tools."""

    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        prompt_builder_service: PromptBuilderService,
        vector_store_component: VectorStoreComponent | None = None,
        node_store_component: NodeStoreComponent | None = None,
        embedding_component: EmbeddingComponent | None = None,
        ingest_component: IngestComponent | None = None,
        parse_component: ParseComponent | None = None,
        content_service: ContentService | None = None,
    ) -> None:
        self.settings = settings
        self.llm_component = llm_component
        self.vector_store_component = vector_store_component
        self.node_store_component = node_store_component
        self.embedding_component = embedding_component
        self.ingest_component = ingest_component
        self.parse_component = parse_component
        self.content_service = content_service
        self.prompt_builder_service = prompt_builder_service

    def _create_text_retriever(self, texts: list[str]) -> InMemoryRetriever:
        """Create an in-memory retriever for text summarization."""
        return InMemoryRetriever.from_texts(texts=texts)

    def _validate_context(self, context_filter: ContextFilter | None) -> ContextFilter:
        if not context_filter:
            raise ValueError("context_filter is required")

        if not context_filter.collection:
            raise ValueError("collection is required in context")

        assert self.vector_store_component, "Vector store component is required"
        assert self.node_store_component, "Node store component is required"
        assert self.embedding_component, "Embedding component is required"
        assert self.ingest_component, "Ingest component is required"
        assert self.parse_component, "Parse component is required"
        assert self.content_service, "Content service is required"

        # If artifacts are provided, verify the related required indexes are ready
        # or throw an error
        if context_filter.artifacts:
            for artifact in context_filter.artifacts:
                vector_artifact_index = VectorArtifactIndex(
                    collection=context_filter.collection,
                    artifact=artifact,
                    vector_store_component=self.vector_store_component,
                    node_store_component=self.node_store_component,
                    embedding_component=self.embedding_component,
                    ingest_component=self.ingest_component,
                    parse_component=self.parse_component,
                )
                vector_artifact_index.populated_or_error()

        return context_filter

    def _create_context_retriever(
        self, context_filter: ContextFilter
    ) -> ContextRetriever:
        """Create a context retriever for knowledge base summarization."""
        self._validate_context(context_filter)
        return ContextRetriever(self.content_service, context_filter)  # type: ignore

    def _create_composite_retriever(
        self,
        texts: list[str] | None = None,
        context_filter: ContextFilter | None = None,
    ) -> "Retriever":
        """Create appropriate retriever(s) based on input parameters."""
        retrievers: list[Retriever] = []

        if texts:
            retrievers.append(self._create_text_retriever(texts))

        if context_filter:
            retrievers.append(self._create_context_retriever(context_filter))

        if not retrievers:
            raise ValueError("Must provide either text or context_filter")

        return CompositeRetriever(retrievers) if len(retrievers) > 1 else retrievers[0]

    def build(
        self,
        texts: list[str] | None = None,
        context_filter: ContextFilter | None = None,
        stop_condition_fn: Callable[[str], Awaitable[bool]] | None = None,
        llm: LLM | None = None,
        timeout: float | None = None,
    ) -> SummarizeWorkflow:
        """Build a summarize workflow."""
        retriever = self._create_composite_retriever(
            texts=texts if texts else None,
            context_filter=context_filter if context_filter else None,
        )

        return SummarizeWorkflow(
            settings=self.settings,
            llm_component=self.llm_component,
            retriever=retriever,
            prompt_builder_service=self.prompt_builder_service,
            stop_condition_fn=stop_condition_fn,
            timeout=timeout,
        )

    async def build_tool(
        self,
        context_filter: ContextFilter | None,
        llm: LLM | None = None,
        name: str = SUMMARIZE_TOOL_NAME,
        tool_type: str = SUMMARIZE_TOOL_NAME + "_v1",
        description: str = SUMMARIZE_TOOL_FN.metadata.description,
        validate: ToolValidationMode = ToolValidationMode.LAZY,
        runtime: Literal["client", "server"] = "server",
        **kwargs: Any,
    ) -> ToolSpec:
        """Builds a summary tool."""
        lock: asyncio.Lock = asyncio.Lock()
        workflow: SummarizeWorkflow | None = None

        async def _ensure_workflow() -> SummarizeWorkflow:
            nonlocal workflow
            if not workflow:
                async with lock:
                    workflow = self.build(
                        context_filter=context_filter,
                        llm=llm,
                        **kwargs,
                    )

            return workflow

        def _sync_format_results(
            content: str | None,
        ) -> list[ResultContentBlockType]:
            return from_tool_output(content)

        async def summarize(
            prompt: str | None = None,
            instructions: str | None = None,
            additional_instructions: list[str] | None = None,
            output_cls: type[BaseModel] | None = None,
        ) -> list[ResultContentBlockType]:
            w = await _ensure_workflow()
            result: SummarizeResultEvent = await w.run(
                start_event=SummarizeInputEvent(
                    prompt=prompt,
                    instructions=instructions,
                    additional_instructions=additional_instructions,
                    output_cls=output_cls,
                )
            )
            assert isinstance(result.result.summary, str), (
                "Expected summary to be a string, "
                f"got {type(result.result.summary)} instead"
            )
            return await asyncio.to_thread(
                _sync_format_results,
                result.result.summary,
            )

        if validate == ToolValidationMode.EAGER and not workflow:
            await _ensure_workflow()

        return ToolSpec.from_defaults(
            name=name,
            type=tool_type,
            runtime=runtime,
            description=description,
            async_fn=summarize,
        )
