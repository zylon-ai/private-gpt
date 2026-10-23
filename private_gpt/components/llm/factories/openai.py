from llama_index.core.llms import LLM

from private_gpt.components.llm.factories.base import LLMFactory
from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase
from private_gpt.settings.settings import LLMModelConfig


class OpenAILLMFactory(LLMFactory):
    """Main OpenAI factory.

    Routes to the Responses API factory when ``model_config.api_type == "responses"``,
    otherwise delegates to the Chat Completions factory.
    """

    def _create_llm(
        self, model_config: LLMModelConfig, tokenizer: TokenizerBase | None = None
    ) -> tuple[LLM, str | None]:
        if model_config.api_type == "responses":
            from private_gpt.components.llm.factories.responses.generic import (
                OpenAIGenericResponsesFactory,
            )

            return OpenAIGenericResponsesFactory(self.settings)._create_llm(
                model_config,
                tokenizer=tokenizer,
            )

        from private_gpt.components.llm.factories.completions.generic import (
            OpenAIGenericCompletionsFactory,
        )

        return OpenAIGenericCompletionsFactory(self.settings)._create_llm(
            model_config,
            tokenizer=tokenizer,
        )
