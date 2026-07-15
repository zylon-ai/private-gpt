from typing import Any

from llama_index.core.llms import LLM

from private_gpt.components.llm.factories.base import LLMFactory
from private_gpt.components.llm.prompt_helper import get_prompt_style
from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase
from private_gpt.settings.settings import LLMModelConfig, Settings
from private_gpt.utils.dependencies import format_missing_dependency_message


def _load_patched_openai_like() -> type[Any]:
    try:
        from private_gpt.components.llm.custom.openailike import PatchedOpenAILikeLLM

        return PatchedOpenAILikeLLM
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message(
                "OpenAI-like LLM",
                extras="llm-openai-compatible",
            )
        ) from e


class OpenAILikeCompletionsFactory(LLMFactory):
    """Chat Completions factory for OpenAI-compatible endpoints (vLLM, Ollama, etc.)."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.openai_settings = settings.openai

    def _create_llm(
        self, model_config: LLMModelConfig, tokenizer: TokenizerBase | None = None
    ) -> tuple[LLM, str | None]:
        PatchedOpenAILikeLLM = _load_patched_openai_like()

        prompt_style = (
            get_prompt_style(
                model_config.prompt_style or "default", tokenizer=tokenizer
            )
            if tokenizer
            else None
        )

        llm = PatchedOpenAILikeLLM(
            api_base=self.openai_settings.api_base,
            api_key=self.openai_settings.api_key or "default",
            model=model_config.name,
            is_chat_model=True,
            api_version="",
            temperature=model_config.sampling_params.temperature or 0.1,
            context_window=model_config.context_window or 3900,
            max_tokens=model_config.sampling_params.max_new_tokens or 1024,
            messages_to_prompt=prompt_style.messages_to_prompt if prompt_style else None,
            completion_to_prompt=prompt_style.completion_to_prompt if prompt_style else None,
            tokenizer=model_config.tokenizer,
            timeout=120.0,
            reuse_client=False,
            logprobs=False,
            is_function_calling_model=bool(model_config.support_tools),
        )

        return llm, model_config.alias
