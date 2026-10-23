from typing import Any, cast

from llama_index.core.llms import LLM

from private_gpt.components.llm.factories.base import LLMFactory
from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase
from private_gpt.settings.settings import LLMModelConfig, Settings
from private_gpt.utils.dependencies import format_missing_dependency_message


def _load_patched_openai() -> type[Any]:
    try:
        from private_gpt.components.llm.custom.openai import PatchedOpenAILLM

        return PatchedOpenAILLM
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message(
                "OpenAI LLM",
                extras="llm-openai",
            )
        ) from e


class OpenAICompletionsFactory(LLMFactory):
    """Chat Completions factory for the real OpenAI endpoint (api.openai.com)."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.openai_settings = settings.openai

    def _create_llm(
        self, model_config: LLMModelConfig, tokenizer: TokenizerBase | None = None
    ) -> tuple[LLM, str | None]:
        PatchedOpenAILLM = _load_patched_openai()

        llm = cast(
            LLM,
            PatchedOpenAILLM(
                api_base=self.openai_settings.api_base or "https://api.openai.com/v1",
                api_key=self.openai_settings.api_key,
                model=model_config.name,
            ),
        )

        return llm, model_config.alias
