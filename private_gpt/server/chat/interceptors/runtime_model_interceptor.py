from injector import inject, singleton

from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import InterceptorPhase
from private_gpt.components.llm.custom.base import ZylonLLM
from private_gpt.components.llm.llm_component import LLMComponent


@singleton
class RuntimeModelRequestInterceptor(ChatRequestLoopInterceptor):
    @inject
    def __init__(self, llm_component: LLMComponent) -> None:
        self._llm_component = llm_component

    async def intercept(self, context: ChatInterceptorContext) -> None:
        if context.phase not in {
            InterceptorPhase.VALIDATION,
            InterceptorPhase.BEFORE_ITERATION,
        }:
            return

        runtime = context.state.runtime
        if runtime.model_id is None:
            runtime.model_id = context.state.input.request.system.model

        if (
            runtime.effective_token_limit is not None
            and runtime.tokenizer_fn is not None
        ):
            return

        llm = self._llm_component.get_llm(runtime.model_id)
        metadata = (
            llm.get_metadata(**context.state.input.llm_kwargs)
            if isinstance(llm, ZylonLLM)
            else llm.metadata
        )
        runtime.effective_token_limit = self._token_limit(
            llm.metadata.context_window,
            metadata.num_output,
        )
        try:
            runtime.tokenizer_fn = self._llm_component.get_tokenizer(runtime.model_id)
        except ValueError:
            runtime.tokenizer_fn = None
        context.set_state(context.state)

    @staticmethod
    def _token_limit(
        context_window: int | None,
        num_output: int | None,
    ) -> int | None:
        if context_window is None or context_window <= 0:
            return None
        reserved_output = num_output or 0
        effective = context_window - reserved_output - 256
        return effective if effective > 0 else context_window
