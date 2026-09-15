import logging

from injector import inject, singleton
from llama_index.core.llms import LLM

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
from private_gpt.components.llm.llm_helper import (
    max_audios_supported,
    max_images_supported,
    supports_audio,
    supports_images,
)
from private_gpt.server.chat.interceptors.preprocessing_tool_calls import (
    PreprocessingToolCalls,
)
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

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
        """Apply multimodal preprocessing to the current chat history.

        This runs only on the first iteration of a request. The original
        image/audio blocks are replaced by their processed text on that first
        pass, so later iterations do not need to reprocess them. Document and
        citation interceptors still run every iteration and consume the
        updated history.
        """
        if (
            context.phase != InterceptorPhase.BEFORE_ITERATION
            or context.state.runtime.iteration > 0
        ):
            return

        state = context.state
        tool_calls = PreprocessingToolCalls(
            context,
            tool_name=self._tool_name,
            return_type=self._preprocess_settings.return_type,
            default_error="There was an error during multimodal processing.",
        )

        try:
            image_model, audio_model = self.resolve_multimodal_models(
                state, context.llm
            )
            model_config = self._llm_component.get_config(
                state.input.request.system.model
            )
            max_images = max_images_supported(context.llm, model_config)
            max_audios = max_audios_supported(context.llm, model_config)

            async for response in preprocess_multimodal_history(
                main_llm=context.llm,
                chat_history=state.input.request.messages,
                image_multimodal_llm=image_model,
                audio_multimodal_llm=audio_model,
                max_concurrency=self._preprocess_settings.max_concurrency,
                return_type=self._preprocess_settings.return_type,
                max_images=max_images,
                max_audios=max_audios,
                timeout=self._preprocess_settings.timeout_seconds,
            ):
                processing = response.processing_status
                if processing is not None:
                    if processing.status == "processing":
                        tool_calls.start(processing.type, {"type": processing.type})
                    elif processing.status in {"completed", "failed"}:
                        tool_calls.finish(
                            processing.type,
                            processing.content,
                            is_error=processing.status == "failed",
                            error_detail=processing.error_detail,
                        )

                if response.chat_history is not None:
                    state.input.request.messages = response.chat_history
        except Exception as exc:
            if not tool_calls.has_pending:
                raise
            logger.exception("Multimodal preprocessing failed; reporting as tool error")
            tool_calls.fail_pending(exc)

        state.input.request.messages = tool_calls.append_tool_messages(
            state.input.request.messages
        )
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
