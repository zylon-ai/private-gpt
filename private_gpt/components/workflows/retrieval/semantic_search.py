import asyncio
import logging
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, cast

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from pydantic import Field

from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.components.llm.llm_helper import get_tokenizer
from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.components.workflows.others.condenser import (
    CondenseResultEvent,
    CondenserWorkflow,
)
from private_gpt.components.workflows.retrieval.retrieval import (
    RetrieverResultEvent,
    RetrieverWorkflow,
)
from private_gpt.components.workflows.types import AnyContext
from private_gpt.di import get_global_injector
from private_gpt.events.models import (
    ResultContentBlockType,
    SourceBlock,
    TextBlock,
)

if TYPE_CHECKING:
    from workflows.handler import WorkflowHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SemanticSearchInputEvent(StartEvent):
    query: str = Field(..., description="The query to process.")
    chat_history: list[ChatMessage] = Field(
        default_factory=list, description="Chat history for context."
    )
    token_limit: int | None = Field(
        None, description="The token limit to apply to the retriever."
    )
    use_condense: bool = Field(True, description="Whether to use query condensing.")
    kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional keyword arguments."
    )


class CondenseEvent(Event):
    query: str = Field(..., description="The query to condense.")
    chat_history: list[ChatMessage] = Field(
        default_factory=list, description="Chat history for context."
    )


class RetrieveEvent(Event):
    query: str = Field(..., description="The query to retrieve nodes for.")
    token_limit: int | None = Field(
        None, description="The token limit to apply to the retriever."
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional keyword arguments."
    )


class SemanticSearchResultEvent(StopEvent):
    retrieval: RetrieverResultEvent = Field(..., description="The retrieval result.")
    condense: CondenseResultEvent | None = Field(
        None, description="The condense result."
    )


class SemanticSearchWorkflow(Workflow):
    """A workflow that combines query condensing and retrieval."""

    def __init__(
        self,
        llm: LLM,
        retriever: BaseRetriever,
        node_postprocessors: list[BaseNodePostprocessor] | None = None,
        node_postprocessors_fn: Callable[..., list[BaseNodePostprocessor]]
        | None = None,
        callback_manager: CallbackManager | None = None,
        prompt_builder_service: PromptBuilderService | None = None,
        condenser_workflow: CondenserWorkflow | None = None,
        retriever_workflow: RetrieverWorkflow | None = None,
        timeout: float | None = None,
        context_filter: ContextFilter | None = None,
        generate_citations: bool = False,
    ):
        super().__init__(timeout=timeout)
        self._context_filter = context_filter
        self._generate_citations = generate_citations

        self._condenser_workflow = condenser_workflow or CondenserWorkflow(
            llm=llm,
            prompt_builder_service=prompt_builder_service,
            callback_manager=callback_manager,
            timeout=timeout,
        )

        self._retriever_workflow = retriever_workflow or RetrieverWorkflow(
            retriever=retriever,
            node_postprocessors=node_postprocessors,
            node_postprocessors_fn=node_postprocessors_fn,
            callback_manager=callback_manager,
            timeout=timeout,
        )

        self._callback_manager = callback_manager or CallbackManager([])

    async def run_semantic_search(
        self,
        query: str,
        use_condense: bool = True,
        generate_citations: bool = False,
        token_limit: int | None = None,
        **kwargs: Any,
    ) -> list[ResultContentBlockType]:
        """Run the semantic search workflow."""

        async def semantic_search() -> list[NodeWithScore]:
            handler: WorkflowHandler | None = None
            try:
                result: SemanticSearchResultEvent = await self.run(
                    start_event=SemanticSearchInputEvent(
                        query=query,
                        use_condense=use_condense,
                        token_limit=token_limit,
                        kwargs=kwargs,
                    )
                )
                return result.retrieval.nodes
            except asyncio.CancelledError as e:
                if handler:
                    await handler.cancel_run()
                raise e

        def format_result(
            n: list[NodeWithScore],
        ) -> str:
            prompt_builder_service = get_global_injector().get(PromptBuilderService)
            prompt, _ = prompt_builder_service.create_context_prompt(
                nodes=n,
                generate_citations=generate_citations,
                token_limit=token_limit,
                tokenizer_fn=get_tokenizer(),
            )
            return prompt.format() or "No content is available."

        nodes = await semantic_search()
        if not nodes:
            return []

        formated_result = await asyncio.to_thread(format_result, nodes)
        return [
            SourceBlock.from_nodes(nodes),
            TextBlock(text=formated_result),
        ]

    @step
    async def start(
        self,
        ctx: AnyContext,
        ev: SemanticSearchInputEvent,
    ) -> CondenseEvent | RetrieveEvent:
        """First step: determine whether to condense the query."""
        await ctx.store.set("original_query", ev.query)
        await ctx.store.set("token_limit", ev.token_limit)
        await ctx.store.set("kwargs", ev.kwargs)

        if ev.use_condense and ev.chat_history:
            logger.info(f"Condensing query: {ev.query}")
            return CondenseEvent(
                query=ev.query,
                chat_history=ev.chat_history,
            )

        return RetrieveEvent(
            query=ev.query, token_limit=ev.token_limit, kwargs=ev.kwargs
        )

    @step
    async def condense(
        self,
        ctx: AnyContext,
        ev: CondenseEvent,
    ) -> RetrieveEvent:
        """Second step: condense the query using the condenser workflow."""
        from private_gpt.components.workflows.others.condenser import CondenseInputEvent

        condense_input = CondenseInputEvent(
            query=ev.query,
            chat_history=ev.chat_history,
        )

        condense_result = cast(
            CondenseResultEvent,
            await self._condenser_workflow.run(start_event=condense_input),
        )
        await ctx.store.set("condense_result", condense_result)

        token_limit = await ctx.store.get("token_limit")
        kwargs = await ctx.store.get("kwargs")

        return RetrieveEvent(
            query=str(condense_result.condensed_query),
            token_limit=token_limit,
            kwargs=kwargs,
        )

    @step
    async def retrieve(
        self, ctx: AnyContext, ev: RetrieveEvent
    ) -> SemanticSearchResultEvent:
        """Third step: retrieve nodes using the retriever workflow."""
        from private_gpt.components.workflows.retrieval.retrieval import (
            RetrieverInputEvent,
        )

        retriever_input = RetrieverInputEvent(
            query=ev.query,
            token_limit=ev.token_limit,
            kwargs=ev.kwargs,
        )

        retrieval_result: RetrieverResultEvent = await self._retriever_workflow.run(
            start_event=retriever_input
        )
        condense_result: CondenseResultEvent | None = None
        with suppress(ValueError):
            condense_result = await ctx.store.get("condense_result")

        return SemanticSearchResultEvent(
            retrieval=retrieval_result,
            condense=condense_result,
        )
