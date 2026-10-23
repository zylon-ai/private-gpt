from typing import Any, cast

from llama_index.core.llms import LLM

from private_gpt.components.llm.factories.base import LLMFactory
from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase
from private_gpt.settings.settings import LLMModelConfig, Settings
from private_gpt.utils.dependencies import format_missing_dependency_message


def _load_patched_openai_like_responses() -> type[Any]:
    try:
        from private_gpt.components.llm.custom.openailikeresponses import (
            PatchedOpenAILikeResponsesLLM,
        )

        return PatchedOpenAILikeResponsesLLM
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message(
                "OpenAI Responses LLM",
                extras="llm-openai",
            )
        ) from e


class OpenAILikeResponsesFactory(LLMFactory):
    """Responses API factory for OpenAI-compatible endpoints (vLLM, Ollama, etc.)."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.openai_settings = settings.openai

    def _create_llm(
        self, model_config: LLMModelConfig, tokenizer: TokenizerBase | None = None
    ) -> tuple[LLM, str | None]:
        PatchedOpenAILikeResponsesLLM = _load_patched_openai_like_responses()

        llm = cast(
            LLM,
            PatchedOpenAILikeResponsesLLM(
                model=model_config.name,
                api_key=self.openai_settings.api_key or "default",
                api_base=self.openai_settings.api_base,
                temperature=model_config.sampling_params.temperature or 0.1,
                max_output_tokens=model_config.sampling_params.max_new_tokens or None,
                context_window=model_config.context_window or None,
                is_function_calling_model=bool(model_config.support_tools),
                timeout=float(self.openai_settings.request_timeout or 120),
            ),
        )

        return llm, model_config.alias
