import asyncio
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from llama_index.core import QueryBundle
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import (
    RESPONSE_TYPE,
    AsyncStreamingResponse,
    PydanticResponse,
    Response,
    StreamingResponse,
)
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.indices.prompt_helper import (
    DEFAULT_CHUNK_OVERLAP_RATIO,
    PromptHelper,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.mixin import PromptMixinType
from llama_index.core.response_synthesizers import (
    BaseSynthesizer,
    ResponseMode,
    get_response_synthesizer,
)
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from llama_index.core.settings import Settings
from pydantic import BaseModel

from private_gpt.components.llm.custom.base import ZylonLLM
from private_gpt.components.workflows.others.summary_retriever import Retriever
from private_gpt.components.workflows.others.tree_summarize_synthesizer import (
    TreeSummarizeSynthesizer,
)
from private_gpt.utils.concurrency import (
    map_elements_in_parallel,
)


class SummaryQueryEngine(BaseQueryEngine):
    def __init__(
        self,
        retriever: Retriever,
        response_synthesizer: BaseSynthesizer | None = None,
        node_postprocessors: list[BaseNodePostprocessor] | None = None,
        callback_manager: CallbackManager | None = None,
        use_async: bool = False,
        max_workers: int | None = None,
        stop_condition_fn: Callable[[str], bool] | None = None,
        async_stop_condition_fn: Callable[[str], Awaitable[bool]] | None = None,
        **response_synthesizer_kwargs: Any,
    ) -> None:
        self._retriever = retriever
        self._response_synthesizer = response_synthesizer or get_response_synthesizer(
            response_mode=ResponseMode.SIMPLE_SUMMARIZE,
            llm=Settings.llm,
            callback_manager=callback_manager or Settings.callback_manager,
        )

        self._node_postprocessors = node_postprocessors or []
        callback_manager = (
            callback_manager or self._response_synthesizer.callback_manager
        )
        for node_postprocessor in self._node_postprocessors:
            node_postprocessor.callback_manager = callback_manager
        self._use_async = use_async
        self._num_workers = max_workers if self._use_async else 1
        self._stop_condition_fn = stop_condition_fn
        self._async_stop_condition_fn = async_stop_condition_fn
        self._response_synthesizer_kwargs = response_synthesizer_kwargs
        super().__init__(callback_manager=callback_manager)

    @classmethod
    def from_args(
        cls,
        retriever: Retriever,
        llm: LLM | None = None,
        callback_manager: CallbackManager | None = None,
        summary_template: BasePromptTemplate | None = None,
        output_cls: type[BaseModel] | None = None,
        stop_condition_fn: Callable[[str], bool] | None = None,
        async_stop_condition_fn: Callable[[str], Awaitable[bool]] | None = None,
        use_async: bool = False,
        max_workers: int | None = None,
        streaming: bool = False,
        verbose: bool = False,
        **response_synthesizer_kwargs: Any,
    ) -> "SummaryQueryEngine":
        llm = llm or Settings.llm
        prompt_helper = SummaryQueryEngine._get_prompt_helper(
            llm=llm,
            **response_synthesizer_kwargs,
        )
        response_synthesizer = TreeSummarizeSynthesizer(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            summary_template=summary_template,
            output_cls=output_cls,
            streaming=streaming,
            use_async=use_async,
            max_workers=max_workers,
            verbose=verbose,
        )
        callback_manager = callback_manager or Settings.callback_manager
        return cls(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            stop_condition_fn=stop_condition_fn,
            async_stop_condition_fn=async_stop_condition_fn,
            callback_manager=callback_manager,
            use_async=use_async,
            max_workers=max_workers,
            **response_synthesizer_kwargs,
        )

    @staticmethod
    def _get_prompt_helper(
        llm: LLM,
        chunk_overlap_ratio: float = DEFAULT_CHUNK_OVERLAP_RATIO,
        chunk_size_limit: int | None = None,
        tokenizer: Callable[[str], list[str]] | None = None,
        separator: str = " ",
        **response_synthesizer_kwargs: Any,
    ) -> PromptHelper:

        llm_metadata = (
            llm.get_metadata(**response_synthesizer_kwargs)
            if isinstance(llm, ZylonLLM)
            else llm.metadata
        )
        context_window: int = llm_metadata.context_window or llm.metadata.context_window
        num_output: int = llm_metadata.num_output or llm.metadata.num_output

        return PromptHelper(
            context_window=context_window,
            num_output=num_output,
            chunk_overlap_ratio=chunk_overlap_ratio,
            chunk_size_limit=chunk_size_limit,
            tokenizer=tokenizer,
            separator=separator,
        )

    def _get_prompt_modules(self) -> PromptMixinType:
        """Get prompt sub-modules."""
        return {
            "response_synthesizer": self._response_synthesizer,
        }

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            if self._use_async:

                async def aprocess_nodes() -> list[BaseNode]:
                    nodes_gen = await self._retriever.aretriever(query_bundle)
                    results: list[BaseNode] = []

                    async for node in map_elements_in_parallel(
                        nodes_gen,
                        lambda n: self.agenerate_summary_nodes(query_bundle, n),
                        num_workers=self._num_workers,
                    ):
                        if node:
                            results.append(node)
                    return results

                # Run async code in sync context
                partial_summary_nodes = asyncio.run(aprocess_nodes())
            else:
                # Synchronous processing remains unchanged
                partial_summary_nodes = []
                for node in self._retriever.retrieve(query_bundle):
                    partial_summary = self.generate_summary_nodes(query_bundle, node)
                    if partial_summary:
                        partial_summary_nodes.append(partial_summary)

            response: RESPONSE_TYPE | None = None
            source_nodes = [
                NodeWithScore(node=node, score=1.0) for node in partial_summary_nodes
            ]

            # Check stop condition
            if self._stop_condition_fn:
                full_content = "\n".join(
                    node.get_content() for node in partial_summary_nodes
                )
                if self._stop_condition_fn(full_content):
                    response = Response(
                        response=full_content,
                        source_nodes=source_nodes,
                    )

            # Generate final response
            if response is None:
                response = self.synthesize(
                    query_bundle=query_bundle,
                    nodes=source_nodes,
                )

            if response is None:
                raise ValueError(
                    "No response was generated. Ensure the retriever returns nodes."
                )

            query_event.on_end(payload={EventPayload.RESPONSE: response})
            return response

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        with self.callback_manager.event(
            CBEventType.QUERY, payload={EventPayload.QUERY_STR: query_bundle.query_str}
        ) as query_event:
            nodes_gen = await self._retriever.aretriever(query_bundle)
            partial_summary_nodes: list[BaseNode] = []

            async for node in map_elements_in_parallel(
                nodes_gen,
                lambda n: self.agenerate_summary_nodes(query_bundle, n),
                num_workers=self._num_workers,
            ):
                if node:
                    partial_summary_nodes.append(node)

            response: RESPONSE_TYPE | None = None
            source_nodes = [
                NodeWithScore(node=node, score=1.0) for node in partial_summary_nodes
            ]

            # Check stop condition
            if self._async_stop_condition_fn:
                full_content = "\n".join(
                    node.get_content() for node in partial_summary_nodes
                )
                result = await self._async_stop_condition_fn(full_content)
                if result:
                    response = Response(
                        response=full_content,
                        source_nodes=source_nodes,
                    )

            # Generate final response
            if response is None:
                response = await self.asynthesize(
                    query_bundle=query_bundle,
                    nodes=source_nodes,
                )

            if response is None:
                raise ValueError(
                    "No response was generated. Ensure the retriever returns nodes."
                )

            query_event.on_end(payload={EventPayload.RESPONSE: response})
            return response

    def generate_summary_nodes(
        self, query_bundle: QueryBundle, node: NodeWithScore
    ) -> BaseNode | None:
        response = self.synthesize(
            query_bundle=query_bundle,
            nodes=[node],
        )
        partial_summary = self.get_response(response)
        if not partial_summary:
            return None
        partial_summary_node = TextNode(**node.dict())
        partial_summary_node.set_content(partial_summary)
        return partial_summary_node

    async def agenerate_summary_nodes(
        self, query_bundle: QueryBundle, node: NodeWithScore
    ) -> BaseNode | None:
        response = await self.asynthesize(
            query_bundle=query_bundle,
            nodes=[node],
        )
        partial_summary = await self.aget_response(response)
        if not partial_summary:
            return None
        partial_summary_node = TextNode(**node.dict())
        partial_summary_node.set_content(partial_summary)
        return partial_summary_node

    def synthesize(
        self,
        query_bundle: QueryBundle,
        nodes: list[NodeWithScore],
        additional_source_nodes: Sequence[NodeWithScore] | None = None,
    ) -> RESPONSE_TYPE:
        return self._response_synthesizer.synthesize(  # type: ignore
            query=query_bundle,
            nodes=nodes,
            **self._response_synthesizer_kwargs,
        )

    async def asynthesize(
        self,
        query_bundle: QueryBundle,
        nodes: list[NodeWithScore],
        additional_source_nodes: Sequence[NodeWithScore] | None = None,
    ) -> RESPONSE_TYPE:
        return await self._response_synthesizer.asynthesize(  # type: ignore
            query=query_bundle,
            nodes=nodes,
            **self._response_synthesizer_kwargs,
        )

    def get_response(
        self,
        response: RESPONSE_TYPE,
    ) -> str | None:
        if isinstance(response, Response):
            return response.response
        elif isinstance(response, PydanticResponse):
            return response.response.model_dump_json() if response.response else None
        elif isinstance(response, StreamingResponse):
            result = ""
            for text in response.response_gen:
                result += text
            return result
        else:
            raise TypeError(f"The result is not of a supported type: {type(response)}")

    async def aget_response(
        self,
        response: RESPONSE_TYPE,
    ) -> str | None:

        if isinstance(response, Response):
            return response.response
        elif isinstance(response, PydanticResponse):
            return response.response.model_dump_json() if response.response else None
        elif isinstance(response, AsyncStreamingResponse):
            result = ""
            async for text in response.response_gen:
                result += text
            return result
        else:
            raise TypeError(f"The result is not of a supported type: {type(response)}")
