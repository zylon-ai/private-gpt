from typing import TYPE_CHECKING
from uuid import uuid4

from injector import inject, singleton
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms import LLM
from llama_index.core.llms.llm import ToolSelection

from private_gpt.components.chat.processors.chat_history.multimodality.multimodality_preprocessor import (
    preprocess_multimodal_history,
)
from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import (
    InterceptorPhase,
)
from private_gpt.components.engines.chat.models.chat_state import (
    ChatState,
)
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.llm.llm_helper import supports_audio, supports_images
from private_gpt.events.models import (
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    ToolResultBlock,
    ToolUseBlock,
    to_llama_index_blocks,
)
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.events.models import ResultContentBlockType

MULTIMODAL_TOOL_NAME = "multimodal_preprocessing"


@singleton
class MultimodalRequestInterceptor(ChatRequestLoopInterceptor):
    """Preprocess image and audio content in conversation history."""

    @inject
    def __init__(self, llm_component: LLMComponent, settings: Settings) -> None:
        self._llm_component = llm_component
        self._tool_name = MULTIMODAL_TOOL_NAME
        self._preprocess_settings = settings.chat.preprocess.multimodal

    async def intercept(self, context: ChatInterceptorContext) -> None:
        """Apply multimodal preprocessing to the current chat history."""
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        state = context.state
        image_model, audio_model = self.resolve_multimodal_models(state, context.llm)

        tool_ids: dict[str, str] = {}
        completed_tools: list[tuple[str, str | list[ResultContentBlockType], bool]] = []

        async for response in preprocess_multimodal_history(
            main_llm=context.llm,
            chat_history=state.input.request.messages,
            image_multimodal_llm=image_model,
            audio_multimodal_llm=audio_model,
            max_concurrency=self._preprocess_settings.max_concurrency,
            return_type=self._preprocess_settings.return_type,
        ):
            processing = response.processing_status
            if processing is not None:
                if processing.status == "processing":
                    tool_id = f"tool_{uuid4().hex}"
                    tool_ids[processing.type] = tool_id
                    use_start = RawContentBlockStartEvent(
                        block_id=f"block_{uuid4().hex}",
                        content_block=ToolUseBlock(
                            id=tool_id,
                            name=self._tool_name,
                            input={"type": processing.type},
                        ),
                    )
                    context.emit_event(use_start)
                    context.emit_event(RawContentBlockStopEvent.from_start(use_start))
                elif processing.status in {"completed", "failed"}:
                    tool_id = tool_ids.get(processing.type, f"tool_{uuid4().hex}")
                    content: str | list[ResultContentBlockType] = (
                        processing.content
                        or processing.error_detail
                        or "There was an error during multimodal processing."
                    )
                    result_start = RawContentBlockStartEvent(
                        block_id=f"block_{uuid4().hex}",
                        content_block=ToolResultBlock(
                            tool_use_id=tool_id,
                            content=content,
                            is_error=processing.status == "failed",
                        ),
                    )
                    context.emit_event(result_start)
                    context.emit_event(
                        RawContentBlockStopEvent.from_start(result_start)
                    )
                    if self._preprocess_settings.return_type == "tool_result":
                        completed_tools.append(
                            (tool_id, content, processing.status == "failed")
                        )

            if response.chat_history is not None:
                state.input.request.messages = response.chat_history

        if self._preprocess_settings.return_type == "tool_result" and completed_tools:
            assistant_msg = ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                additional_kwargs={
                    "tool_calls": [
                        ToolSelection(
                            tool_id=tool_id,
                            tool_name=self._tool_name,
                            tool_kwargs={},
                        )
                        for tool_id, _, _ in completed_tools
                    ]
                },
            )
            tool_msgs: list[ChatMessage] = []
            for tool_id, content, _ in completed_tools:
                kwargs = {
                    "tool_call_id": tool_id,
                    "tool_call_name": self._tool_name,
                    "raw_output": content,
                }
                if isinstance(content, str):
                    tool_msgs.append(
                        ChatMessage(
                            role=MessageRole.TOOL,
                            content=content,
                            additional_kwargs=kwargs,
                        )
                    )
                else:
                    tool_msgs.append(
                        ChatMessage(
                            role=MessageRole.TOOL,
                            blocks=to_llama_index_blocks(content),
                            additional_kwargs=kwargs,
                        )
                    )
            state.input.request.messages = [
                *state.input.request.messages,
                assistant_msg,
                *tool_msgs,
            ]

        context.set_state(state)

    def resolve_multimodal_models(
        self,
        state: ChatState,
        main_llm: LLM,
    ) -> tuple[LLM | None, LLM | None]:
        """Resolve optional multimodal models using configured LLM registry."""
        request = state.input.request
        image_model: LLM | None = None
        audio_model: LLM | None = None

        model_id = request.system.model
        model_config = self._llm_component.get_config(model_id)

        if supports_images(main_llm, model_config):
            image_model = main_llm
        elif not model_id:
            potential = next(
                self._llm_component.filter(
                    lambda potential_llm, cfg: supports_images(potential_llm, cfg)
                ),
                None,
            )
            if potential is not None:
                image_model = potential[0]

        if supports_audio(main_llm, model_config):
            audio_model = main_llm
        elif not model_id:
            potential = next(
                self._llm_component.filter(
                    lambda potential_llm, cfg: supports_audio(potential_llm, cfg)
                ),
                None,
            )
            if potential is not None:
                audio_model = potential[0]

        return image_model, audio_model
