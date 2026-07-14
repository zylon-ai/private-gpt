import asyncio
from collections.abc import AsyncGenerator
from typing import Any

from injector import inject, singleton
from pydantic import BaseModel, Field, ValidationError

from private_gpt.chat.extensions.citation import ZylonCitation
from private_gpt.chat.input_models import CountTokensOutput
from private_gpt.components.chat.models.chat_config_models import (
    ChatRequest,
    ResolvedChatRequest,
)
from private_gpt.components.chunk.models import SourceType
from private_gpt.components.container_registry import ContainerRegistry
from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.engines.chat.async_chat_engine import (
    AsyncChatEngine,
)
from private_gpt.components.engines.chat.chat_engine import ChatLoopEngine
from private_gpt.components.engines.chat.chat_engine_interface import (
    ChatEngine,
    LoopChatEngineAdapter,
)
from private_gpt.components.engines.chat.models.execution_hooks import (
    ExecutionHooks,
)
from private_gpt.components.engines.chat.resumable_runner import ResumableChatRunner
from private_gpt.components.ingest.ingest_component import IngestComponent
from private_gpt.components.llm.custom.base import ZylonLLM
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.llm.models import ReasoningEffort
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.streaming.tasks.chat_scheduler import ChatSchedulerFactory
from private_gpt.components.tools.tool_scheduler import ToolSchedulerFactory
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.events.event_errors import Errors
from private_gpt.events.event_folding import fold
from private_gpt.events.models import (
    ContentBlockType,
    Event,
    SourceBlock,
    ToolResultBlock,
    Usage,
)
from private_gpt.events.models import (
    TextBlock as OutTextBlock,
)
from private_gpt.server.chat.interceptors.chat_interceptor_service import (
    ChatInterceptorService,
)
from private_gpt.server.models.models_service import ModelsService
from private_gpt.settings.settings import Settings
from private_gpt.utils.tokens import estimate_token_count


class Completion(BaseModel):
    content: list[ContentBlockType] = Field(
        default_factory=list, description="Content blocks"
    )
    exception: BaseException | None = Field(
        default=None, description="Exception if any"
    )
    stop_reason: str | None = Field(default=None, description="Finish reason")
    usage: Usage = Field(description="Usage information")

    @property
    def response(self) -> str | None:
        """Get the response from the content blocks."""
        if not self.content:
            return None
        text_block = [
            content for content in self.content if isinstance(content, OutTextBlock)
        ]
        return text_block[-1].text if text_block else None

    @property
    def sources(self) -> list[SourceType] | None:
        """Get the sources from the content blocks."""
        if not self.content:
            return None

        potential_sources = [
            source
            for content in self.content
            if isinstance(content, ToolResultBlock)
            if isinstance(content.content, list)
            for source_block in content.content
            if isinstance(source_block, SourceBlock)
            for source in source_block.sources
        ]

        unique_sources: dict[str | None, SourceType] = {
            source.id: source for source in potential_sources
        }
        return list(unique_sources.values()) if unique_sources else None

    @property
    def citations(self) -> list[ZylonCitation] | None:
        """Get the citations from the content blocks."""
        if not self.content:
            return None
        return [
            citation
            for content in self.content
            if isinstance(content, OutTextBlock)
            for citation in content.citations or []
        ]

    class Config:
        arbitrary_types_allowed = True

        @staticmethod
        def json_schema_extra(
            schema: dict[str, Any], model: type["Completion"]
        ) -> None:
            props = schema.get("properties", {})
            for prop_name in ["response", "sources", "citations"]:
                if prop_name in props:
                    del props[prop_name]


class CompletionGen(BaseModel):
    events: AsyncGenerator[Event, None]
    final_state_task: Any | None = None

    class Config:
        arbitrary_types_allowed = True


class ChatValidationResult(BaseModel):
    """Result of chat request validation."""

    valid: bool = Field(default=False, description="Is the request valid")
    errors: list[str] | None = Field(
        default=None, description="List of validation errors if any"
    )

    class Config:
        arbitrary_types_allowed = True


@singleton
class ChatService:
    settings: Settings
    llm_component: LLMComponent
    vector_store_component: VectorStoreComponent
    embedding_component: EmbeddingComponent

    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
        ingest_component: IngestComponent,
        chat_interceptor_service: ChatInterceptorService,
        models_service: ModelsService,
        container_registry: ContainerRegistry,
        scheduler_factory: ToolSchedulerFactory,
        chat_scheduler_factory: ChatSchedulerFactory,
        resumable_runner: ResumableChatRunner,
    ) -> None:
        self.settings = settings
        self.llm_component = llm_component
        self.vector_store_component = vector_store_component
        self.node_store_component = node_store_component
        self.embedding_component = embedding_component
        self.ingest_component = ingest_component
        self.chat_interceptor_service = chat_interceptor_service
        self.models_service = models_service
        self.container_registry = container_registry
        self._tool_scheduler = scheduler_factory.get()
        self._chat_scheduler = chat_scheduler_factory.get()
        self._resumable_runner = resumable_runner

    def build_async_engine(self) -> AsyncChatEngine:
        # Don't build a singleton since the interceptors
        # and state can be mutated during the loop execution
        # for several threads handling different requests in parallel
        chain = self.chat_interceptor_service.get_chain()
        return AsyncChatEngine(
            llm_component=self.llm_component,
            request_interceptors=chain.request_interceptors,
            response_interceptors=chain.response_interceptors,
            tool_interceptors=chain.tool_interceptors,
            max_iterations=40,
            container_registry=self.container_registry,
            tool_scheduler=self._tool_scheduler,
            chat_scheduler=self._chat_scheduler,
            resumable_runner=self._resumable_runner,
        )

    def build_loop_engine(self) -> LoopChatEngineAdapter:
        chain = self.chat_interceptor_service.get_chain()
        return LoopChatEngineAdapter(
            engine=ChatLoopEngine(
                llm_component=self.llm_component,
                request_interceptors=chain.request_interceptors,
                response_interceptors=chain.response_interceptors,
                tool_interceptors=chain.tool_interceptors,
                max_iterations=40,
                container_registry=self.container_registry,
                tool_scheduler=self._tool_scheduler,
            )
        )

    def build_engine(self) -> ChatEngine:
        if self.settings.chat.engine_mode == "async":
            return self.build_async_engine()
        return self.build_loop_engine()

    async def stream_chat(
        self,
        request: ChatRequest,
        hooks: ExecutionHooks | None = None,
    ) -> CompletionGen:
        engine = await asyncio.to_thread(self.build_engine)
        execution = await engine.run(request=request, hooks=hooks)
        return CompletionGen(
            events=execution.events,
            final_state_task=execution.final_state_task,
        )

    async def chat(
        self,
        request: ChatRequest,
        hooks: ExecutionHooks | None = None,
    ) -> Completion:
        completion_gen = await self.stream_chat(request, hooks=hooks)
        wrapped_response = await fold(completion_gen.events)
        usage = Usage.model_validate(wrapped_response.usage or {})
        return Completion(
            content=wrapped_response.content,
            exception=wrapped_response.exception,
            stop_reason=wrapped_response.stop_reason,
            usage=usage,
        )

    async def cancel(self, correlation_id: str) -> bool:
        engine = await asyncio.to_thread(self.build_engine)
        return await engine.cancel(correlation_id=correlation_id)

    async def validate(self, request: ResolvedChatRequest) -> ChatValidationResult:
        errors: list[str] = []

        try:
            engine = await asyncio.to_thread(self.build_engine)
            await engine.validate(request=request)
        except ValidationError as e:
            for err in e.errors():
                field = " -> ".join(str(loc) for loc in err["loc"])
                errors.append(f"{field}: {err['msg']}")
        except ValueError as e:
            errors.append(str(e))
        except Errors.Base as e:
            errors.append(str(e))
        except Exception as e:
            wrapped_error = Errors.build(e)
            errors.append(wrapped_error.error_type)

        return ChatValidationResult(valid=len(errors) == 0, errors=errors or None)

    async def count_tokens(self, request: ResolvedChatRequest) -> CountTokensOutput:
        model_id = request.system.model or None
        llm = self.llm_component.get_llm(model_id)

        try:
            tokenizer_fn = self.llm_component.get_tokenizer(model_id)
        except Exception as e:
            raise ValueError(f"Model '{model_id}' does not support tokenization") from e

        chat_messages = request.to_messages()

        input_tokens = await estimate_token_count(
            chat_history=chat_messages,
            tools=request.tool_config.tools,
            reasoning_effort=(
                ReasoningEffort.from_str(request.thinking.type)
                if request.thinking.enabled and request.thinking.type
                else ReasoningEffort.NONE
            ),
            message_to_input=(
                llm.message_to_input
                if isinstance(llm, ZylonLLM)
                else llm.messages_to_prompt
            ),
            tokenizer_fn=tokenizer_fn,
        )
        return CountTokensOutput(input_tokens=input_tokens)
