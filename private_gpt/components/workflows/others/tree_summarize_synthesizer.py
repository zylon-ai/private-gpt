from collections.abc import Coroutine, Sequence
from typing import Any

from llama_index.core.async_utils import asyncio_run, run_async_tasks
from llama_index.core.bridge.pydantic import BaseModel
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.llms import LLM
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.prompts.default_prompt_selectors import (
    DEFAULT_TREE_SUMMARIZE_PROMPT_SEL,
)
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.types import RESPONSE_TEXT_TYPE

from private_gpt.utils.concurrency import bounded_concurrent_execute


def run_async_tasks_with_limit_concurrency(
    tasks: list[Coroutine[Any, Any, Any]],
    num_workers: int,
) -> list[Any]:
    outputs: list[Any] = asyncio_run(bounded_concurrent_execute(tasks, num_workers))
    return outputs


class TreeSummarizeSynthesizer(BaseSynthesizer):
    """Tree summarize response builder.

    Code taken from Llama Index, modifying the class
    to support lazy code and allow to limit concurrency.

    This response builder recursively merges text chunks and summarizes them
    in a bottom-up fashion (i.e. building a tree from leaves to root).

    More concretely, at each recursively step:
    1. we repack the text chunks so that each chunk fills the context window of the LLM
    2. if there is only one chunk, we give the final response
    3. otherwise, we summarize each chunk and recursively summarize the summaries.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        callback_manager: CallbackManager | None = None,
        prompt_helper: PromptHelper | None = None,
        summary_template: BasePromptTemplate | None = None,
        output_cls: type[BaseModel] | None = None,
        streaming: bool = False,
        use_async: bool = False,
        max_workers: int | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            streaming=streaming,
            output_cls=output_cls,
        )
        self._summary_template = summary_template or DEFAULT_TREE_SUMMARIZE_PROMPT_SEL
        self._use_async = use_async
        self._num_workers = max_workers
        self._verbose = verbose

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {"summary_template": self._summary_template}

    def _update_prompts(self, prompts_dict: PromptDictType) -> None:
        """Update prompts."""
        if "summary_template" in prompts_dict:
            self._summary_template = prompts_dict["summary_template"]

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get tree summarize response."""
        summary_template = self._summary_template.partial_format(query_str=query_str)
        # repack text_chunks so that each chunk fills the context window
        text_chunks = self._prompt_helper.repack(
            summary_template, text_chunks=text_chunks, llm=self._llm
        )

        if self._verbose:
            print(f"{len(text_chunks)} text chunks after repacking")

        # give final response if there is only one chunk
        if len(text_chunks) == 1:
            response: RESPONSE_TEXT_TYPE
            if self._streaming:
                response = await self._llm.astream(
                    summary_template, context_str=text_chunks[0], **response_kwargs
                )
            else:
                if self._output_cls is None:
                    response = await self._llm.apredict(
                        summary_template,
                        context_str=text_chunks[0],
                        **response_kwargs,
                    )
                else:
                    response = await self._llm.astructured_predict(
                        self._output_cls,
                        summary_template,
                        context_str=text_chunks[0],
                        **response_kwargs,
                    )

            # return pydantic object if output_cls is specified
            return response

        else:
            # summarize each chunk
            tasks: list[Coroutine[Any, Any, Any]]
            if self._output_cls is None:
                tasks = [
                    self._llm.apredict(
                        summary_template,
                        context_str=text_chunk,
                        **response_kwargs,
                    )
                    for text_chunk in text_chunks
                ]
            else:
                tasks = [
                    self._llm.astructured_predict(
                        self._output_cls,
                        summary_template,
                        context_str=text_chunk,
                        **response_kwargs,
                    )
                    for text_chunk in text_chunks
                ]

            summary_responses = await bounded_concurrent_execute(
                tasks,
                concurrency_limit=self._num_workers,
                jitter=(0.0, 10.0),
            )
            if self._output_cls is not None:
                summaries = [summary.model_dump_json() for summary in summary_responses]
            else:
                summaries = summary_responses

            # recursively summarize the summaries
            return await self.aget_response(
                query_str=query_str,
                text_chunks=summaries,
                **response_kwargs,
            )

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Get tree summarize response."""
        summary_template = self._summary_template.partial_format(query_str=query_str)
        # repack text_chunks so that each chunk fills the context window
        text_chunks = self._prompt_helper.repack(
            summary_template, text_chunks=text_chunks, llm=self._llm
        )

        if self._verbose:
            print(f"{len(text_chunks)} text chunks after repacking")

        # give final response if there is only one chunk
        if len(text_chunks) == 1:
            response: RESPONSE_TEXT_TYPE
            if self._streaming:
                response = self._llm.stream(
                    summary_template, context_str=text_chunks[0], **response_kwargs
                )
            else:
                if self._output_cls is None:
                    response = self._llm.predict(
                        summary_template,
                        context_str=text_chunks[0],
                        **response_kwargs,
                    )
                else:
                    response = self._llm.structured_predict(
                        self._output_cls,
                        summary_template,
                        context_str=text_chunks[0],
                        **response_kwargs,
                    )

            return response

        else:
            # summarize each chunk
            if self._use_async:
                tasks: list[Coroutine[Any, Any, Any]]
                if self._output_cls is None:
                    tasks = [
                        self._llm.apredict(
                            summary_template,
                            context_str=text_chunk,
                            **response_kwargs,
                        )
                        for text_chunk in text_chunks
                    ]
                else:
                    tasks = [
                        self._llm.astructured_predict(
                            self._output_cls,
                            summary_template,
                            context_str=text_chunk,
                            **response_kwargs,
                        )
                        for text_chunk in text_chunks
                    ]

                summary_responses = (
                    run_async_tasks_with_limit_concurrency(tasks, self._num_workers)
                    if self._num_workers is not None
                    else run_async_tasks(tasks)
                )

                if self._output_cls is not None:
                    summaries = [
                        summary.model_dump_json() for summary in summary_responses
                    ]
                else:
                    summaries = summary_responses
            else:
                if self._output_cls is None:
                    summaries = [
                        self._llm.predict(
                            summary_template,
                            context_str=text_chunk,
                            **response_kwargs,
                        )
                        for text_chunk in text_chunks
                    ]
                else:
                    summaries = [
                        self._llm.structured_predict(
                            self._output_cls,
                            summary_template,
                            context_str=text_chunk,
                            **response_kwargs,
                        )
                        for text_chunk in text_chunks
                    ]
                    summaries = [summary.model_dump_json() for summary in summaries]

            # recursively summarize the summaries
            return self.get_response(
                query_str=query_str, text_chunks=summaries, **response_kwargs
            )
