import asyncio
import logging
from collections.abc import Awaitable, Callable, Iterator
from typing import TYPE_CHECKING, Any

from llama_index.core import BasePromptTemplate, ChatPromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.base.response.schema import (
    PydanticResponse,
    Response,
    StreamingResponse,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from pydantic import BaseModel, Field, SkipValidation
from workflows.handler import WorkflowHandler

from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.llm.priorities import DefinedPriorities
from private_gpt.components.markdown.markdown_helper import MarkdownHelper
from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.components.workflows.others.summary_query_engine import (
    SummaryQueryEngine,
)
from private_gpt.components.workflows.others.summary_retriever import (
    Retriever,
)
from private_gpt.events.models import TextBlock
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from workflows.handler import WorkflowHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SummarizeInputEvent(StartEvent):
    model_id: str | None = Field(default=None, description="Model identifier to use")
    prompt: str | None = Field(default=None, description="System prompt override")
    instructions: str | None = Field(
        default=None, description="Instructions for summarization"
    )
    additional_instructions: list[str] | None = Field(
        default=None, description="Additional instructions"
    )
    stream: bool = Field(
        default=False,
        description="Whether the summarization stream is enabled",
    )
    output_cls: SkipValidation[type[BaseModel]] | None = Field(
        default=None,
        description="Optional output class for structured results",
    )
    empty_response_fallback: str | None = Field(
        default=None,
        description="Optional fallback response for empty response",
    )


class SummarizeResultEvent(StopEvent):
    summary: str | None = Field(default=None, description="The generated summary text")
    output_obj: BaseModel | None = Field(
        default=None,
        description="Optional output object if output_cls was provided",
    )


class SummarizeWorkflow(Workflow):
    """Async workflow for document/text summarization."""

    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        retriever: Retriever,
        prompt_builder_service: PromptBuilderService,
        stop_condition_fn: Callable[[str], Awaitable[bool]] | None = None,
        callback_manager: CallbackManager | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ):
        super().__init__(timeout=timeout)
        self.settings = settings
        self.llm_component = llm_component
        self.retriever = retriever
        self.prompt_builder_service = prompt_builder_service
        self.stop_condition_fn = stop_condition_fn

        # Set callback manager for LLM if provided
        if callback_manager:
            self.llm_component.llm.callback_manager = callback_manager

    async def run_summary(
        self,
        model_id: str | None = None,
        prompt: str | None = None,
        instructions: str | None = None,
        additional_instructions: list[str] | None = None,
        output_cls: type[BaseModel] | None = None,
        empty_response_fallback: str | None = None,
        **kwargs: Any,
    ) -> list[TextBlock]:
        """Run the summarization workflow and return formatted content blocks."""
        handler: WorkflowHandler | None = None
        try:
            handler = self.run(
                start_event=SummarizeInputEvent(
                    model_id=model_id,
                    prompt=prompt,
                    instructions=instructions,
                    additional_instructions=additional_instructions,
                    output_cls=output_cls,
                    empty_response_fallback=empty_response_fallback,
                )
            )
            result: SummarizeResultEvent = await handler

            if output_cls:
                response = result.output_obj
                if not response:
                    raise ValueError("No output object was generated")
                if not isinstance(response, BaseModel):
                    raise TypeError(
                        f"Expected output object to be a BaseModel, got {type(response)}"
                    )
                return [TextBlock(text=response.model_dump_json())]
            else:
                summary = result.summary
                summary_text = summary if isinstance(summary, str) else None

                if not summary_text:
                    raise ValueError("No summary was generated")

                return [TextBlock(text=summary_text)]
        except asyncio.CancelledError as e:
            if handler:
                await handler.cancel_run()
            raise e

    async def _generate_prompt_template(
        self,
        prompt: str | None = None,
    ) -> BasePromptTemplate:
        """Define the prompt template for summarization."""

        def messages_gen() -> Iterator[ChatMessage]:
            if prompt:
                yield ChatMessage(
                    content=prompt,
                    role=MessageRole.SYSTEM,
                )

            yield ChatMessage(
                content=(
                    "Context information from multiple sources is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the information from multiple sources and not prior knowledge, "
                    "answer the query.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                ),
                role=MessageRole.USER,
            )

        return ChatPromptTemplate(
            message_templates=list(messages_gen()),
        )

    @step
    async def execute_summarize(self, ev: SummarizeInputEvent) -> SummarizeResultEvent:
        # Configure token limits
        max_new_tokens = self.llm_component.metadata(ev.model_id).num_output
        max_tokens = max(4000, max_new_tokens * 4)

        llm = self.llm_component.get_llm(ev.model_id)
        tokenizer = self.llm_component.get_tokenizer(ev.model_id)

        # Create query engine with the provided retriever
        query_engine = SummaryQueryEngine.from_args(
            retriever=self.retriever,
            streaming=ev.stream,
            # Configure LLM and tokenizer
            llm=llm,
            tokenizer=tokenizer,
            max_workers=self.settings.server.max_workers,
            priority=DefinedPriorities.LLM.SUMMARY_PRIORITY,
            max_tokens=max_tokens,
            output_cls=ev.output_cls,
            summary_template=await self._generate_prompt_template(
                prompt=ev.prompt,
            ),
            async_stop_condition_fn=self.stop_condition_fn,
            empty_response=(
                ev.empty_response_fallback if ev.output_cls is None else None
            ),
        )

        # Build prompt template
        template = self.prompt_builder_service.create_summary_prompt(
            user_query=ev.instructions,
            additional_instructions="\n".join(ev.additional_instructions or []),
            max_words=int(max_tokens * 0.75),
        )

        logger.debug(f"Executing summarization with max_tokens: {max_tokens}")
        task = asyncio.create_task(query_engine.aquery(template.format()))
        try:
            response = await task
        except asyncio.CancelledError:
            logger.info("Summarization task was cancelled")
            task.cancel()
            raise

        logger.debug("Summarization completed successfully")

        if ev.output_cls and isinstance(response, PydanticResponse):
            if not response.response:
                raise ValueError("No response was generated")

            return SummarizeResultEvent(
                output_obj=response.response,
            )

        if isinstance(response, Response):
            summary = response.response or ev.empty_response_fallback or ""
            if not summary:
                raise ValueError("No summary was generated")

            sanitized = MarkdownHelper.sanitize_markdown(summary)
            return SummarizeResultEvent(summary=sanitized or summary)

        elif isinstance(response, StreamingResponse):
            raise NotImplementedError(
                "Streaming responses are not yet implemented for summarization"
            )

        raise TypeError(f"Unsupported response type: {type(response)}")
