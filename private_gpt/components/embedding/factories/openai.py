from llama_index.core.base.embeddings.base import BaseEmbedding

from private_gpt.components.embedding.factories.base import EmbeddingFactory
from private_gpt.components.model_discovery.url_utils import is_openai_api_base
from private_gpt.settings.settings import EmbeddingModelConfig, Settings
from private_gpt.utils.dependencies import format_missing_dependency_message


class OpenAIGPTEmbeddingFactory(EmbeddingFactory):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def _create_embedding(
        self, model_config: EmbeddingModelConfig
    ) -> tuple[BaseEmbedding, str | None]:
        try:
            from llama_index.embeddings.openai import (  # ty:ignore[unresolved-import]
                OpenAIEmbedding,
            )
        except ImportError as e:
            raise ImportError(
                format_missing_dependency_message(
                    "OpenAI embeddings",
                    extras="embedding-openai",
                )
            ) from e

        api_base = (
            self.settings.openai.embedding_api_base or self.settings.openai.api_base
        )
        api_key = self.settings.openai.embedding_api_key or self.settings.openai.api_key
        model = model_config.name

        embedding_model = OpenAIEmbedding(
            api_base=api_base,
            api_key=api_key,
            model=model,
        )
        return embedding_model, model_config.name


class OpenAIEmbeddingFactory(EmbeddingFactory):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def _create_embedding(
        self, model_config: EmbeddingModelConfig
    ) -> tuple[BaseEmbedding, str | None]:
        api_base = (
            self.settings.openai.embedding_api_base or self.settings.openai.api_base
        )
        if is_openai_api_base(api_base):
            return OpenAIGPTEmbeddingFactory(self.settings)._create_embedding(
                model_config
            )

        from private_gpt.components.embedding.factories.openailike import (
            OpenAILikeEmbeddingFactory,
        )

        return OpenAILikeEmbeddingFactory(self.settings)._create_embedding(model_config)
