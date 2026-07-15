import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeGuard

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, QueryType
from llama_index.core.tools import ToolOutput
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from pydantic import BaseModel, Field
from workflows.retry_policy import ConstantDelayRetryPolicy

from private_gpt.components.engines.citations.utils import (
    exclude_metadata,
    init_nodes_with_shorter_ids,
)
from private_gpt.components.workflows.types import AnyContext

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


_MAX_RETRIES = 5
_JITTER = (5.0, 15.0)
RETRY_POLICY = ConstantDelayRetryPolicy(delay=_JITTER[0], maximum_attempts=_MAX_RETRIES)


def _is_node_postprocessor_list(
    value: object,
) -> TypeGuard[list[BaseNodePostprocessor]]:
    return isinstance(value, list) and all(
        isinstance(item, BaseNodePostprocessor) for item in value
    )


class RetrieverConfig(BaseModel):
    """Configuration for the RetrieverWorkflow."""

    init_short_ids: bool = Field(
        default=True,
        description="Whether to initialize nodes with shorter IDs.",
    )
    exclude_metadata: bool = Field(
        default=True,
        description="Whether to exclude metadata from the nodes.",
    )


class RetrieverInputEvent(StartEvent):
    """Event to start the retriever workflow."""

    query: QueryType = Field(..., description="The query to retrieve nodes for.")
    token_limit: int | None = Field(
        None, description="The token limit to apply to the retriever."
    )
    config: RetrieverConfig = Field(
        default_factory=RetrieverConfig,
        description="Configuration for the retriever workflow.",
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional keyword arguments."
    )


class RawNodesRetrievedEvent(Event):
    """Event containing raw retrieved nodes before transformation."""

    nodes: list[NodeWithScore] = Field(..., description="The raw nodes retrieved.")


class FinalNodesRetrievalEvent(Event):
    """Event containing transformed nodes."""

    nodes: list[NodeWithScore] = Field(..., description="The transformed nodes.")


class RetrieverResultEvent(StopEvent):
    """Event indicating retrieval workflow completion."""

    nodes: list[NodeWithScore] = Field(description="The final nodes retrieved.")
    source: ToolOutput = Field(description="The source of the nodes.")


class RetrievalProgressEvent(Event):
    """Event for streaming progress updates."""

    stage: str = Field(..., description="The stage of the retrieval process.")
    message: str = Field(..., description="The message to display.")
    count: int | None = Field(default=None, description="The count of nodes processed.")


class RetrieverWorkflow(Workflow):
    """A workflow that retrieves nodes from a retriever and applies postprocessing."""

    def __init__(
        self,
        retriever: BaseRetriever,
        node_postprocessors: list[BaseNodePostprocessor] | None = None,
        node_postprocessors_fn: Callable[
            ..., list[BaseNodePostprocessor] | Awaitable[list[BaseNodePostprocessor]]
        ]
        | None = None,
        callback_manager: CallbackManager | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ):
        """Initialize the RetrieverWorkflow."""
        super().__init__(timeout=timeout)
        self._retriever = retriever
        self._node_postprocessors = node_postprocessors or []
        self._node_postprocessors_fn = node_postprocessors_fn
        self._callback_manager = callback_manager or CallbackManager([])

        # Set the callback manager for the retriever
        if self._callback_manager:
            self._retriever.callback_manager = self._callback_manager

    @step(retry_policy=RETRY_POLICY)
    async def retrieve_raw_nodes(
        self, ctx: AnyContext, ev: RetrieverInputEvent
    ) -> RawNodesRetrievedEvent | FinalNodesRetrievalEvent:
        """Step 1: Retrieve raw nodes from the retriever."""
        await ctx.store.set("query", ev.query)
        await ctx.store.set("config", ev.config or RetrieverConfig())
        await ctx.store.set("token_limit", ev.token_limit)
        await ctx.store.set("kwargs", ev.kwargs)

        ctx.write_event_to_stream(
            RetrievalProgressEvent(
                stage="retrieve", message=f"Starting retrieval for: {ev.query}"
            )
        )

        logger.debug(f"Retrieving nodes for query: {ev.query}")
        nodes: list[NodeWithScore] = await self._retriever.aretrieve(str(ev.query))

        node_count = len(nodes)
        logger.debug(f"Retrieved {node_count} raw nodes")

        if node_count == 0:
            return FinalNodesRetrievalEvent(
                nodes=nodes,
            )

        ctx.write_event_to_stream(
            RetrievalProgressEvent(
                stage="retrieve",
                message=f"Retrieved {node_count} nodes",
                count=node_count,
            )
        )

        # Pass the retrieved nodes to the next step
        return RawNodesRetrievedEvent(
            nodes=nodes,
        )

    @step(retry_policy=RETRY_POLICY)
    async def transform_nodes(
        self, ctx: AnyContext, ev: RawNodesRetrievedEvent
    ) -> FinalNodesRetrievalEvent:
        """Step 2: Transform the raw nodes through filtering and post-processing."""
        logger.debug(f"Transforming {len(ev.nodes)} raw nodes")

        query: QueryType | None = await ctx.store.get("query")
        config: RetrieverConfig = await ctx.store.get("config")
        token_limit: int | None = await ctx.store.get("token_limit")

        # Apply node postprocessors
        logger.debug(f"Transforming {len(ev.nodes)} raw nodes")
        nodes: list[NodeWithScore] = ev.nodes
        node_processors = await self._get_node_postprocessors(
            query=query, token_limit=token_limit
        )
        if node_processors:
            ctx.write_event_to_stream(
                RetrievalProgressEvent(
                    stage="transform",
                    message=f"Filtering and expanding {len(nodes)} nodes",
                )
            )
            query_bundle = (
                QueryBundle(query_str=str(query))
                if not isinstance(query, QueryBundle)
                else query
            )
            for i, postprocessor in enumerate(node_processors):
                logger.debug(
                    f"Applying postprocessor: {postprocessor.__class__.__name__} ({i+1}/{len(node_processors)})"
                )
                nodes = await postprocessor.apostprocess_nodes(
                    nodes, query_bundle=query_bundle
                )
                logger.debug(
                    f"Postprocessor {postprocessor.__class__.__name__} applied, {len(nodes)} nodes remaining"
                )

        # Shorten node IDs and exclude metadata
        def process_nodes_async(
            nodes: list[NodeWithScore],
        ) -> list[NodeWithScore]:
            nodes = init_nodes_with_shorter_ids(nodes)
            nodes = exclude_metadata(nodes)
            return nodes

        if config:
            nodes = await asyncio.to_thread(process_nodes_async, nodes)

        ctx.write_event_to_stream(
            RetrievalProgressEvent(
                stage="transform",
                message=f"Transformed to {len(nodes)} nodes",
                count=len(nodes),
            )
        )

        logger.debug(f"Transformed to {len(nodes)} nodes")
        return FinalNodesRetrievalEvent(nodes=nodes)

    @step
    async def finalize_nodes(
        self, ctx: AnyContext, ev: FinalNodesRetrievalEvent
    ) -> RetrieverResultEvent:
        """Step 3: Finalize the retrieval process."""
        logger.debug(f"Finalizing {len(ev.nodes)} nodes")

        query: QueryType | None = await ctx.store.get("query")

        source = ToolOutput(
            tool_name="retriever",
            content=f"Retrieved {len(ev.nodes)} nodes"
            if ev.nodes
            else "No relevant nodes found.",
            raw_input={"message": query},
            raw_output=ev.nodes,
        )

        if ev.nodes:
            ctx.write_event_to_stream(
                RetrievalProgressEvent(
                    stage="finalize", message=f"Finalized {len(ev.nodes)} nodes"
                )
            )

        logger.debug(f"Finalized {len(ev.nodes)} nodes for query: {query}")
        return RetrieverResultEvent(nodes=ev.nodes, source=source)

    async def _get_node_postprocessors(
        self, **kwargs: Any
    ) -> list[BaseNodePostprocessor]:
        node_postprocessors: list[
            BaseNodePostprocessor
        ] | None = self._node_postprocessors
        if self._node_postprocessors_fn:
            if asyncio.iscoroutinefunction(self._node_postprocessors_fn):
                result = await self._node_postprocessors_fn(**kwargs)
            else:
                result = await asyncio.to_thread(self._node_postprocessors_fn, **kwargs)

            if inspect.isawaitable(result):
                result = await result
            if not _is_node_postprocessor_list(result):
                raise TypeError("Node postprocessor callback must return a list")

            node_postprocessors = result

        if node_postprocessors is None:
            return []

        if not isinstance(node_postprocessors, list):
            node_postprocessors = list(node_postprocessors)

        for postprocessor in node_postprocessors:
            postprocessor.callback_manager = self._callback_manager

        return node_postprocessors
