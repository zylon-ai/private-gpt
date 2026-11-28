from typing import cast

from llama_index.core.llms import LLM

from private_gpt.components.llm.factories.base import LLMFactory
from private_gpt.components.llm.factories.responses.generic import (
    OpenAIResponsesLLMLoader,
)
from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase
from private_gpt.settings.settings import LLMModelConfig, Settings


class OpenAIResponsesFactory(OpenAIResponsesLLMLoader, LLMFactory):
    """Responses API factory for the real OpenAI endpoint (api.openai.com)."""

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.openai_settings = settings.openai

    def _create_llm(
        self, model_config: LLMModelConfig, tokenizer: TokenizerBase | None = None
    ) -> tuple[LLM, str | None]:
        PatchedOpenAIResponsesLLM = self._load_patched_openai_responses()
        sampling_params = model_config.sampling_params

        llm = cast(
            LLM,
            PatchedOpenAIResponsesLLM(
                model=model_config.name,
                api_key=self.openai_settings.api_key,
                api_base=self.openai_settings.api_base or "https://api.openai.com/v1",
                temperature=sampling_params.temperature,
                # Only cap the output when the user configured a limit.
                max_output_tokens=sampling_params.max_new_tokens
                if "max_new_tokens" in sampling_params.model_fields_set
                else None,
            ),
        )

        return llm, model_config.alias
