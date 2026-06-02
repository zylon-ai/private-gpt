import logging

from injector import inject, singleton

from private_gpt.chat.input_models import PromptConfig
from private_gpt.components.engines.chat_loop.interceptors.chat_loop_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_interceptor_context import (
    ChatLoopInterceptorContext,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_phase import (
    InterceptorPhase,
)
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class DefaultValuesRequestInterceptor(ChatRequestLoopInterceptor):
    @inject
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def intercept(self, context: ChatLoopInterceptorContext) -> None:
        if context.phase != InterceptorPhase.VALIDATION:
            return

        state = context.state
        request = state.input.request

        # Configure system prompt — use_default_prompt is deprecated.
        # When set to True it enables all platform prompts for backwards compatibility.
        if request.system.use_default_prompt:
            logger.warning(
                "use_default_prompt is deprecated. Use system.prompt to "
                "control per-category prompt injection."
            )
            request.system.platform_prompts = PromptConfig(
                tools=True,
                citations=True,
                thinking=True,
            )
        request.system.use_default_prompt = (
            request.system.use_default_prompt
            and self._settings.chat.allow_use_default_prompt
        )

        # Configure context
        request.context.add_context_to_system_prompt = (
            request.context.add_context_to_system_prompt
            or self._settings.chat.add_context_to_system_prompt
        )
        request.context.deduplicate_context_in_history = (
            request.context.deduplicate_context_in_history
            or self._settings.chat.deduplicate_context_in_history
        )
        request.context.maximum_context_length = (
            request.context.maximum_context_length
            or self._settings.chat.maximum_context_length
            or None
        )

        # Configure citations
        request.citation.enabled = (
            request.citation.enabled and self._settings.chat.allow_generate_citations
        )
        request.citation.force_to_return_citations = (
            request.citation.force_to_return_citations
            or self._settings.chat.force_to_return_citations
        )
        request.citation.return_missing_citations = (
            request.citation.return_missing_citations
            or self._settings.chat.return_missing_citations
        )

        # Configure reasoning
        request.thinking.enabled = (
            request.thinking.enabled and self._settings.chat.allow_reasoning
        )

        state.input.request = request
        context.set_state(state)
