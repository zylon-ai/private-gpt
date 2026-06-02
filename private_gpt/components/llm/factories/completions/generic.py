from llama_index.core.llms import LLM

from private_gpt.components.llm.factories.base import LLMFactory
from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase
from private_gpt.components.model_discovery.url_utils import is_openai_api_base
from private_gpt.settings.settings import LLMModelConfig, Settings


class OpenAIGenericCompletionsFactory(LLMFactory):
    """Chat Completions router.

    Routes to OpenAICompletionsFactory for api.openai.com,
    otherwise delegates to OpenAILikeCompletionsFactory.
    """

    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.openai_settings = settings.openai

    def _create_llm(
        self, model_config: LLMModelConfig, tokenizer: TokenizerBase | None = None
    ) -> tuple[LLM, str | None]:
        if is_openai_api_base(self.openai_settings.api_base):
            from private_gpt.components.llm.factories.completions.openai import (
                OpenAICompletionsFactory,
            )

            return OpenAICompletionsFactory(self.settings)._create_llm(
                model_config,
                tokenizer=tokenizer,
            )

        from private_gpt.components.llm.factories.completions.openailike import (
            OpenAILikeCompletionsFactory,
        )

        return OpenAILikeCompletionsFactory(self.settings)._create_llm(
            model_config,
            tokenizer=tokenizer,
        )
