import asyncio
import logging
import time
from asyncio import CancelledError
from typing import TYPE_CHECKING, Any

from llama_index.core import PromptTemplate
from pydantic import BaseModel

from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.tools.builders.summary_builder import (
    SummarizeWorkflowBuilder,
)
from private_gpt.components.web.web_scraper_service import (
    WebScraperService,
)
from private_gpt.components.web.web_search.models import WebSearchResult
from private_gpt.components.web.web_search.processors.base import (
    BaseWebSearchResultProcessor,
)
from private_gpt.settings.settings import (
    Settings,
    WebSearchParams,
    settings,
)
from private_gpt.utils.tokens import async_tokenizer

if TYPE_CHECKING:
    from asyncio import Task

    from private_gpt.components.web.web_scraper_service import (
        WebScraperResult,
    )

debug_mode = settings().server.debug_mode
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)


class RelevanceOutput(BaseModel):
    relevant: bool


MAX_TOKEN_RATIO = 0.9


class SelectBestLinks(BaseWebSearchResultProcessor):
    _scraper_service: WebScraperService
    _llm_component: LLMComponent
    _summary_builder: SummarizeWorkflowBuilder
    _params: WebSearchParams = WebSearchParams()
    _max_tokens: int = 0

    def __init__(
        self,
        settings: Settings,
        scraper_service: WebScraperService,
        llm_component: LLMComponent,
        summary_builder: SummarizeWorkflowBuilder,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._scraper_service = scraper_service
        self._llm_component = llm_component
        self._summary_builder = summary_builder
        self._init_params()

    async def validate(self) -> None:
        if not self._scraper_service.is_initialized:
            raise ValueError(
                "Web fetching is not properly initialized or it is disabled. "
                "Since Web Search depends on web fetching to retrieve content, "
                "the Web Search functionality cannot operate correctly. "
                "Consider enabling web fetching in settings."
            )

    async def process_results(
        self,
        query: str,
        results: list[WebSearchResult],
        model_id: str | None = None,
    ) -> list[WebSearchResult]:
        start = time.perf_counter()
        max_tokens = int(
            (
                self._settings.web_search.context_token
                or self._llm_component.metadata(model_id).context_window
            )
            * MAX_TOKEN_RATIO
        )
        token_limit = max_tokens / self._params.num_references

        scrape_queue: asyncio.Queue[WebSearchResult | None] = asyncio.Queue()
        summary_queue: asyncio.PriorityQueue[tuple[float, WebSearchResult | None]] = (
            asyncio.PriorityQueue()
        )
        is_finished_queue: asyncio.Queue[asyncio.Event | None] = asyncio.Queue()

        scraped_results: list[WebSearchResult] = []
        final_selection: list[WebSearchResult] | None = None

        for res in results[: self._settings.web_search.num_links]:
            await scrape_queue.put(res)
            scraped_results.append(res)

        for _ in range(self._params.num_concurrent_consumers):
            await scrape_queue.put(None)

        async def scrape_worker() -> None:
            while True:
                res = await scrape_queue.get()
                if res is None:
                    break
                try:
                    await self._scrape_url(res)
                    if not res.is_in_error:
                        is_relevance = (
                            await self._validate_relevance(query, res, model_id)
                            or False
                        )
                        if is_relevance:
                            copy = WebSearchResult.model_validate(res.model_dump())
                            copy.is_relevant = True
                            copy.tokens = len(
                                await async_tokenizer(
                                    res.content or "",
                                    tokenizer_fn=self._llm_component.get_tokenizer(
                                        model_id
                                    ),
                                )
                            )
                            scraped_results[scraped_results.index(res)] = copy

                            priority = self._compute_priority(copy, max_tokens)
                            await summary_queue.put((priority, copy))
                except Exception as e:
                    logger.debug(f"Error processing {res.url}: {e}")
                    res.is_in_error = True
                finally:
                    await is_finished_queue.put(asyncio.Event())

        async def summary_worker() -> None:
            while True:
                _priority, res = await summary_queue.get()
                if res is None:
                    break

                if res.tokens > token_limit and res.tokens < max_tokens:
                    logger.debug(f"Reached {res.url}, summarizing to reduce tokens")
                    content = await self._summarize_result(res, model_id)
                    if content:
                        copy = WebSearchResult.model_validate(res.model_dump())
                        copy.content = content
                        copy.tokens = len(
                            await async_tokenizer(
                                content,
                                tokenizer_fn=self._llm_component.get_tokenizer(
                                    model_id
                                ),
                            )
                        )
                        scraped_results[scraped_results.index(res)] = copy

                    await is_finished_queue.put(asyncio.Event())
                else:
                    logger.debug(f"Skipping summary for {res.url} - within token limit")

        async def check_selection() -> list[WebSearchResult] | None:
            if (
                sum(1 for r in scraped_results if not r.is_in_error and r.is_relevant)
                < self._params.num_references
            ):
                return None

            selected: list[WebSearchResult] = []
            total_tokens = 0

            # Case 1: Prioritize index
            if self._params.index_weight and self._params.index_weight > 0:
                valid_results = [res for res in scraped_results if not res.is_in_error]
                sorted_results = sorted(valid_results, key=lambda x: x.idx or -1)[
                    : self._params.num_references
                ]

            # Case 2: Prioritize token efficiency
            else:
                valid_results = [
                    res
                    for res in scraped_results
                    if not res.is_in_error and res.is_relevant and res.content
                ]
                sorted_results = sorted(valid_results, key=lambda x: x.tokens)

            if len(sorted_results) < self._params.num_references:
                return None  # Not enough valid results yet

            for res in sorted_results:
                if total_tokens + res.tokens > max_tokens:
                    continue
                selected.append(res)
                total_tokens += res.tokens
                if len(selected) >= self._params.num_references:
                    break

            logger.debug(
                f"Current selection: {len(selected)} links, total tokens: {total_tokens}/{max_tokens}"
            )

            if (
                len(selected) >= self._params.num_references
                and total_tokens <= max_tokens
            ):
                logger.debug(
                    f"Find {len(selected)} links with total size {total_tokens}/{max_tokens} "
                )
                return selected
            return None

        scrape_task = [
            asyncio.create_task(scrape_worker())
            for _ in range(self._params.num_concurrent_consumers)
        ]
        summary_tasks = [
            asyncio.create_task(summary_worker())
            for _ in range(
                min(
                    self._params.max_parallel_summary,
                    self._params.num_concurrent_consumers,
                )
            )
        ]

        worker_tasks: list[Task[Any]] = [
            *scrape_task,
            *summary_tasks,
        ]

        async def selection_worker() -> None:
            nonlocal final_selection
            while not final_selection:
                event = await is_finished_queue.get()
                if event is None:
                    break
                result = await check_selection()
                if result:
                    final_selection = result

                    scrape_queue.task_done()
                    summary_queue.task_done()

                    raise CancelledError()

        selections_task = asyncio.create_task(selection_worker())
        all_tasks = [*worker_tasks, selections_task]

        try:
            await asyncio.wait_for(
                selections_task, timeout=self._params.max_timeout_seconds
            )
        except (TimeoutError, CancelledError):
            logger.debug("Completion criteria met or timeout reached")

        for task in all_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)

        # Return results
        total_elapsed = time.perf_counter() - start
        logger.debug(f"⏱️ Total time process_results: {total_elapsed:.3f}s")

        if final_selection:
            logger.debug(f"Returning final selection of {len(final_selection)} links")
            return final_selection
        else:
            logger.debug("No optimal selection found, returning best effort")
            candidates = sorted(
                (r for r in scraped_results if not r.is_in_error and r.is_relevant),
                key=lambda r: r.idx or 0,
            )
            fallback: list[WebSearchResult] = []
            total_tokens = 0
            for r in candidates:
                if total_tokens + r.tokens > max_tokens:
                    continue
                fallback.append(r)
                total_tokens += r.tokens
                if len(fallback) >= self._params.num_references:
                    break
            return fallback

    def _init_params(self) -> None:
        match self._settings.web_search.mode_quality:
            case "accurate":
                self._params = self._settings.web_search.accurate_params
            case "fast":
                self._params = self._settings.web_search.fast_params

        if self._settings.web_search.context_token == 0:
            self._settings.web_search.context_token = None

    def _compute_priority(self, res: WebSearchResult, max_tokens: int) -> float:
        idx = res.idx or 0

        tokens = res.tokens
        normalized_tokens = tokens / max_tokens

        token_exponent_penalty = self._params.token_exponent_penalty or 0.0
        index_weight = self._params.index_weight or 0.0
        token_weight = self._params.token_weight or 0.0

        token_penalty = normalized_tokens**token_exponent_penalty

        priority = float(index_weight * idx + token_weight * token_penalty)
        logger.debug(
            f"Computing priority: index={idx}, tokens={tokens}, max_tokens={max_tokens}, "
            f"normalized_tokens={normalized_tokens:.3f}, penalty={token_penalty:.3f}, priority={priority:.3f}"
        )
        return priority

    async def _scrape_url(self, res: WebSearchResult) -> None:
        start = time.perf_counter()
        try:
            logger.debug(f"Scraping {res.url}")
            response: WebScraperResult = (
                await self._scraper_service.scrape_max_compress(res.url)
            )
            res.content = response.markdown_content
            res.favicon_url = response.favicon_url
            res.content_type = "text/markdown"
            res.is_in_error = False
            logger.debug(f"Scraped {res.url}")
        except asyncio.CancelledError:
            res.is_in_error = True
            raise
        except Exception as e:
            logger.debug(f"Error scraping {res.url}: {e}")
            res.is_in_error = True
        finally:
            elapsed = time.perf_counter() - start
            logger.debug(f"Scraping time {res.url}: {elapsed:.3f}s")

    async def _validate_relevance(
        self, query: str, res: WebSearchResult, model_id: str | None = None
    ) -> bool | None:
        start = time.perf_counter()

        # TODO: Move to jinja2 templates
        prompt_str = (
            "You are an assistant that evaluates whether the provided text contains relevant, factual, and useful "
            "information to answer the question:\n"
            f'"{query}"\n\n'
            "Respond with True or False only."
            f"Text to evaluate:\n{res.content}\n\n"
        )[
            : int(
                self._llm_component.get_config(model_id).context_window
                * MAX_TOKEN_RATIO
            )
        ]

        prompt = PromptTemplate(template=prompt_str)

        try:
            llm = self._llm_component.get_llm(model_id)

            result = await llm.astructured_predict(
                RelevanceOutput,
                prompt,
            )

            logger.debug(f"LLM response: {result}")
            elapsed = time.perf_counter() - start
            logger.debug(f"Validate time {res.url}: {elapsed:.3f}s")

            return result.relevant if hasattr(result, "relevant") else False

        except Exception as e:
            logger.debug(f"Error during async structured relevance validation: {e}")
            return None

    async def _summarize_result(
        self, res: WebSearchResult, model_id: str | None = None
    ) -> str | None:
        """Asynchronously produce a summary and update res.summary."""
        start = time.perf_counter()

        try:
            workflow = await asyncio.to_thread(
                self._summary_builder.build,
                texts=[res.content or ""],
                timeout=self._params.max_summary_timeout_seconds,
            )

            summary_blocks = await workflow.run_summary(
                model_id=model_id,
                prompt=None,
                stream=False,
            )

            summary_content = "\n".join(block.text for block in summary_blocks)
            summary = summary_content.strip() if summary_content else None

            elapsed = time.perf_counter() - start
            logger.debug(f"Summary time {res.url}: {elapsed:.3f}s")

            return summary
        except asyncio.CancelledError:
            logger.debug(f"Summary task for {res.url} was cancelled")
            raise
