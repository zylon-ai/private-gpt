import asyncio
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from pydantic import Field

from private_gpt.components.readers.nodes import TableNode
from private_gpt.components.tabular.pandasai_service import PandasAIService
from private_gpt.components.workflows.retrieval.retrieval import (
    RetrieverConfig,
    RetrieverWorkflow,
)
from private_gpt.components.workflows.types import AnyContext
from private_gpt.events.models import (
    ResultContentBlockType,
    TextBlock,
    from_tool_output,
)
from private_gpt.server.utils.artifact_input import SqlDatabaseArtifact

if TYPE_CHECKING:
    from workflows.handler import WorkflowHandler

    from private_gpt.components.workflows.retrieval.retrieval import (
        RetrieverResultEvent,
    )

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TabularDataAnalysisInputEvent(StartEvent):
    query: str = Field(..., description="The query to process.")
    kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Additional keyword arguments."
    )


class CreatedDataFrame(Event):
    data_frames: list[pd.DataFrame] = Field(
        default_factory=list, description="The created dataframes."
    )
    is_error: bool = Field(
        default=False,
        description="Indicates if there was an error during dataframe creation.",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if there was an error during dataframe creation.",
    )


class DataFramesReadyEvent(Event):
    data_frames: list[pd.DataFrame] = Field(
        default_factory=list, description="The dataframes ready for analysis."
    )
    is_error: bool = Field(
        default=False,
        description="Error message if there was an error during preparation.",
    )
    error_message: str | None = Field(
        None, description="Error message if there was an error during preparation."
    )


class TabularDataAnalysisResultEvent(StopEvent):
    content: list[Any] = Field(
        default_factory=list, description="The content blocks from the analysis."
    )
    is_error: bool = Field(
        False,
        description="Indicates if the analysis resulted in an error.",
    )


class TabularDataAnalysisWorkflow(Workflow):
    """A workflow that combines query condensing and retrieval."""

    def __init__(
        self,
        llm: LLM,
        pandas_ai: PandasAIService,
        retriever: BaseRetriever | None,
        db_connections: list[SqlDatabaseArtifact] | None = None,
        node_postprocessors: list[BaseNodePostprocessor] | None = None,
        node_postprocessors_fn: Callable[..., list[BaseNodePostprocessor]]
        | None = None,
        callback_manager: CallbackManager | None = None,
        retriever_workflow: RetrieverWorkflow | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(timeout=timeout)
        self._workflow_kwargs = kwargs

        self._llm = llm
        self._pandas_ai = pandas_ai
        self._live_db_connections = db_connections or []
        self._retriever_workflow = None
        if retriever:
            self._retriever_workflow = retriever_workflow or RetrieverWorkflow(
                retriever=retriever,
                node_postprocessors=node_postprocessors,
                node_postprocessors_fn=node_postprocessors_fn,
                callback_manager=callback_manager,
                timeout=timeout,
                **kwargs,
            )
        else:
            self._retriever_workflow = None

        self._callback_manager = callback_manager or CallbackManager([])

    async def run_tabular_data_analysis(
        self,
        query: str,
        **kwargs: Any,
    ) -> tuple[list[ResultContentBlockType], bool]:
        """Run the Tabular Data Analysis workflow."""

        async def tabular_data_analysis() -> tuple[list[ResultContentBlockType], bool]:
            handler: WorkflowHandler | None = None
            try:
                result: TabularDataAnalysisResultEvent = await self.run(
                    start_event=TabularDataAnalysisInputEvent(
                        query=query,
                        kwargs=kwargs,
                    )
                )
                return result.content, result.is_error
            except asyncio.CancelledError as e:
                if handler:
                    await handler.cancel_run()
                raise e

        results, is_in_error = await tabular_data_analysis()
        content_blocks: list[ResultContentBlockType] = []
        if results:
            content_blocks = [
                output for result in results for output in from_tool_output(result)
            ]
            unique_results = []
            seen_ids = set()

            for item in content_blocks:
                item_id = id(item)
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    unique_results.append(item)
            content_blocks = unique_results

        if content_blocks and is_in_error:
            return content_blocks, is_in_error
        elif not results:
            return [TextBlock(text="There are no results for the query.")], is_in_error
        elif is_in_error:
            return [
                TextBlock(text="A fatal error occurred during the analysis.")
            ], is_in_error

        return content_blocks, is_in_error

    async def _retrieve_from_nodes(
        self, ctx: AnyContext, ev: TabularDataAnalysisInputEvent
    ) -> CreatedDataFrame:
        from private_gpt.components.workflows.retrieval.retrieval import (
            RetrieverInputEvent,
        )

        retriever_input = RetrieverInputEvent(
            query=ev.query,
            config=RetrieverConfig(
                init_short_ids=False,  # Do not shorten IDs in this workflow
            ),
            token_limit=None,
            kwargs=ev.kwargs,
        )

        assert self._retriever_workflow is not None, (
            "Retriever workflow is not configured."
        )
        retrieval_result: RetrieverResultEvent = await self._retriever_workflow.run(
            start_event=retriever_input
        )

        # Filter nodes (only keep TableNode)
        nodes = [
            node.node
            for node in retrieval_result.nodes
            if isinstance(node.node, TableNode)
        ]
        if not nodes:
            return CreatedDataFrame(
                data_frames=[], is_error=False, error_message="No table data found."
            )

        # Deduplicate nodes
        nodes = list({node.id_: node for node in nodes}.values())

        # Remove spaces and unknown chars in table columns
        for node in nodes:
            if isinstance(node.df, pd.DataFrame):
                node.df.columns = pd.Index(
                    [
                        re.sub(r"[^a-zA-Z0-9_]", "", col.replace(" ", "_"))
                        for col in node.df.columns
                    ]
                )

        # Get all dataframes
        dataframes = [node.df for node in nodes]
        return CreatedDataFrame(data_frames=dataframes)

    @step
    async def retrieve(
        self,
        ctx: AnyContext,
        ev: TabularDataAnalysisInputEvent,
    ) -> DataFramesReadyEvent | None:
        await ctx.store.set("query", ev.query)
        await ctx.store.set("kwargs", ev.kwargs)

        dfs_coro = []
        if self._retriever_workflow:
            dfs_coro.append(self._retrieve_from_nodes(ctx=ctx, ev=ev))

        dfs: list[CreatedDataFrame] = list(await asyncio.gather(*dfs_coro))

        any_failed = any(df.is_error for df in dfs)
        joined_error_messages = "; ".join(
            df.error_message for df in dfs if df.is_error and df.error_message
        )
        # join all the dataframes
        data_frames = [
            df for created in dfs for df in created.data_frames if created.data_frames
        ]
        return DataFramesReadyEvent(
            data_frames=data_frames,
            is_error=any_failed,
            error_message=joined_error_messages if any_failed else None,
        )

    @step
    async def analyze_df(
        self, ctx: AnyContext, ev: DataFramesReadyEvent
    ) -> TabularDataAnalysisResultEvent:
        if ev.is_error:
            return TabularDataAnalysisResultEvent(
                content=[ev.error_message or "Error querying data."], is_error=True
            )

        if not ev.data_frames:
            return TabularDataAnalysisResultEvent(
                content=["No data found."], is_error=False
            )

        # Run the analysis
        query: str = await ctx.store.get("query")
        kwargs: dict[str, Any] = await ctx.store.get("kwargs")
        result = await self._pandas_ai.run_analysis(query, *ev.data_frames, **kwargs)
        return TabularDataAnalysisResultEvent(
            content=result.content, is_error=bool(result.error)
        )
