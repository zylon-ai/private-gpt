from uuid import uuid4

from injector import inject, singleton
from llama_index.core.llms import LLM

from private_gpt.components.chat.processors.chat_history.multimodality.multimodality_preprocessor import (
    preprocess_multimodal_history,
)
from private_gpt.components.engines.chat_loop.interceptors.chat_loop_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_interceptor_context import (
    ChatLoopInterceptorContext,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_phase import (
    InterceptorPhase,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_state import (
    ChatLoopState,
)
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.llm.llm_helper import supports_audio, supports_images
from private_gpt.events.models import (
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    ToolResultBlock,
    ToolUseBlock,
)

MULTIMODAL_TOOL_NAME = "multimodal_preprocessing"


@singleton
class MultimodalRequestInterceptor(ChatRequestLoopInterceptor):
    """Preprocess image and audio content in conversation history."""

    @inject
    def __init__(self, llm_component: LLMComponent) -> None:
        self._llm_component = llm_component
        self._tool_name = MULTIMODAL_TOOL_NAME

    async def intercept(self, context: ChatLoopInterceptorContext) -> None:
        """Apply multimodal preprocessing to the current chat history."""
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        state = context.state
        image_model, audio_model = self.resolve_multimodal_models(state, context.llm)

        async for response in preprocess_multimodal_history(
            main_llm=context.llm,
            chat_history=state.input.request.messages,
            image_multimodal_llm=image_model,
            audio_multimodal_llm=audio_model,
        ):
            processing = response.processing_status
            if processing is not None:
                tool_id = f"tool_{uuid4().hex}"
                if processing.status == "processing":
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
                    result_start = RawContentBlockStartEvent(
                        block_id=f"block_{uuid4().hex}",
                        content_block=ToolResultBlock(
                            tool_use_id=tool_id,
                            content=(
                                processing.content
                                or processing.error_detail
                                or "There was an error during multimodal processing."
                            ),
                            is_error=processing.status == "failed",
                        ),
                    )
                    context.emit_event(result_start)
                    context.emit_event(
                        RawContentBlockStopEvent.from_start(result_start)
                    )

            if response.chat_history is not None:
                state.input.request.messages = response.chat_history

        context.set_state(state)

    def resolve_multimodal_models(
        self,
        state: ChatLoopState,
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
