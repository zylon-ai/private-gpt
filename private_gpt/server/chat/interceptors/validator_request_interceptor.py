from injector import inject, singleton
from llama_index.core.base.llms.types import (
    AudioBlock,
    ChatMessage,
    ImageBlock,
    MessageRole,
    TextBlock,
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
from private_gpt.components.llm.custom.base import ZylonLLM
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.llm.llm_helper import (
    max_audios_supported,
    max_images_supported,
    supports_audio,
    supports_images,
)
from private_gpt.events.event_errors import Errors


@singleton
class ValidatorRequestInterceptor(ChatRequestLoopInterceptor):
    @inject
    def __init__(self, llm_component: LLMComponent) -> None:
        self._llm_component = llm_component

    async def intercept(self, context: ChatInterceptorContext) -> None:
        if context.phase != InterceptorPhase.VALIDATION:
            return

        request = context.state.input.request
        model_id = request.system.model

        llm = self._llm_component.get_llm(model_id)
        model_config = self._llm_component.get_config(model_id)
        model_name = model_id or model_config.alias or model_config.name

        # Validate if the model supports tool usage
        tools = context.state.input.context_stack.all_tools()
        mcp_servers = context.state.input.request.mcp_servers
        needs_tool_support = bool(tools) or bool(mcp_servers)
        if needs_tool_support and not llm.metadata.is_function_calling_model:
            raise Errors.InvalidRequest(
                f"Model '{model_name}' does not support tool usage, but tools are configured",
                Errors.Codes.INVALID_REQUEST_TOOL_SUPPORTED_ERROR,
            )

        # Validate if the model supports the required features
        needs_reasoning = request.thinking.enabled
        if needs_reasoning and not model_config.support_reasoning:
            raise Errors.InvalidRequest(
                f"Model '{model_name}' does not support reasoning, but reasoning is enabled",
                Errors.Codes.INVALID_REQUEST_REASONING_SUPPORTED_ERROR,
            )

        last_user_message = self._last_user_message(
            context.state.input.request.messages
        )
        user_text = self._extract_text(last_user_message)
        if not user_text:
            raise Errors.InvalidRequest(
                "The message cannot be empty.",
                Errors.Codes.INVALID_REQUEST_EMPTY_MESSAGE_ERROR,
            )

        # Validate multimodal inputs (images)
        images: list[ImageBlock] = [
            image for image in last_user_message.blocks if isinstance(image, ImageBlock)
        ]
        if images:
            if not supports_images(llm, model_config):
                raise Errors.InvalidRequest(
                    "The LLM does not support images, but the message contains image blocks.",
                    Errors.Codes.INVALID_REQUEST_IMAGE_SUPPORT_ERROR,
                )
            max_num_images = max_images_supported(llm, model_config)
            if len(images) > max_num_images:
                raise Errors.InvalidRequest(
                    f"The LLM supports a maximum of {max_num_images} images, but the message contains {len(images)}",
                    Errors.Codes.INVALID_REQUEST_IMAGE_MAX_NUM_ERROR,
                )

        # Validate multimodal inputs (audios)
        audios: list[AudioBlock] = [
            audio for audio in last_user_message.blocks if isinstance(audio, AudioBlock)
        ]
        if audios:
            if not supports_audio(llm, model_config):
                raise Errors.InvalidRequest(
                    "The LLM does not support audio, but the message contains audio blocks.",
                    Errors.Codes.INVALID_REQUEST_AUDIO_SUPPORT_ERROR,
                )
            max_num_audios = max_audios_supported(llm, model_config)
            if len(audios) > max_num_audios:
                raise Errors.InvalidRequest(
                    f"The LLM supports a maximum of {max_num_audios} audios, but the message contains {len(audios)}",
                    Errors.Codes.INVALID_REQUEST_AUDIO_MAX_NUM_ERROR,
                )

        metadata = (
            llm.get_metadata(**context.state.input.llm_kwargs)
            if isinstance(llm, ZylonLLM)
            else llm.metadata
        )
        token_limit = self._token_limit(
            llm.metadata.context_window, metadata.num_output
        )
        if token_limit is None:
            return

        tokenize = self._llm_component.get_tokenizer(model_id)
        if tokenize is None:
            return

        user_message_tokens = len(tokenize(user_text))
        if user_message_tokens > token_limit:
            raise Errors.RequestTooLarge(
                f"The message length {user_message_tokens} exceeds the maximum token limit {token_limit}.",
                Errors.Codes.REQUEST_TOO_LARGE_USER_MSG,
            )

        # If a system message is present in the request messages, it's a misuse
        potential_system_message = self._system_message_text(
            context.state.input.request.messages
        )
        if potential_system_message:
            raise RuntimeError(
                "System messages should be as layer in the context stack."
            )

        # Prefer system prompt from the context stack, fall back to prompt
        system_prompt_block = (
            context.state.input.context_stack.to_system_prompt()
            or request.system.get_prompt()
            or None
        )
        system_prompt = (
            "\n".join(
                [block.text for block in system_prompt_block]
                if system_prompt_block
                else []
            )
            if system_prompt_block
            else None
        )
        system_tokens = len(tokenize(system_prompt)) if system_prompt else 0
        if system_tokens > token_limit:
            raise Errors.RequestTooLarge(
                f"The system prompt length {system_tokens} exceeds the maximum token limit {token_limit}.",
                Errors.Codes.REQUEST_TOO_LARGE_SYSTEM_MSG,
            )

        combined = user_message_tokens + system_tokens
        if combined > token_limit:
            raise Errors.RequestTooLarge(
                f"The message length {user_message_tokens} and system prompt length {system_tokens} "
                f"exceed the maximum token limit {token_limit}."
            )

        # Update state with effective token limit for downstream components
        state = context.state
        state.runtime.effective_token_limit = token_limit
        state.runtime.tokenizer_fn = tokenize
        context.set_state(state)

        return

    @staticmethod
    def _token_limit(
        context_window: int | None,
        num_output: int | None,
    ) -> int | None:
        """Compute effective token limit from model metadata."""
        if context_window is None or context_window <= 0:
            return None
        reserved_output = num_output or 0
        effective = context_window - reserved_output - 256
        if effective > 0:
            return effective
        return context_window

    @staticmethod
    def _extract_text(message: ChatMessage) -> str:
        """Extract normalized text from message blocks."""
        parts = [
            block.text.strip()
            for block in message.blocks
            if isinstance(block, TextBlock) and block.text and block.text.strip()
        ]
        return "\n".join(parts)

    @staticmethod
    def _last_user_message(messages: list[ChatMessage]) -> ChatMessage:
        """Return the latest user message from request messages."""
        for message in reversed(messages):
            if message.role == MessageRole.USER:
                return message
        raise Errors.InvalidRequest("No user message found in request.")

    @staticmethod
    def _system_message_text(messages: list[ChatMessage]) -> str | None:
        """Return first system message text when present."""
        for message in messages:
            if message.role != MessageRole.SYSTEM:
                continue
            text = ValidatorRequestInterceptor._extract_text(message)
            if text:
                return text
        return None
