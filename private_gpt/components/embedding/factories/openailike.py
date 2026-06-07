from llama_index.core.base.embeddings.base import BaseEmbedding

from private_gpt.components.embedding.factories.base import EmbeddingFactory
from private_gpt.settings.settings import EmbeddingModelConfig, Settings
from private_gpt.utils.dependencies import format_missing_dependency_message


class OpenAILikeEmbeddingFactory(EmbeddingFactory):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def _create_embedding(
        self, model_config: EmbeddingModelConfig
    ) -> tuple[BaseEmbedding, str | None]:
        try:
            from llama_index.embeddings.openai_like import (  # type: ignore
                OpenAILikeEmbedding,
            )
        except ImportError as e:
            raise ImportError(
                format_missing_dependency_message(
                    "OpenAI-like Embeddings",
                    extras="embedding-openai-compatible",
                )
            ) from e

        api_base = (
            self.settings.openai.embedding_api_base or self.settings.openai.api_base
        )
        api_key = (
            self.settings.openai.embedding_api_key
            or self.settings.openai.api_key
            or "default"
        )
        model = model_config.name

        embedding_model = OpenAILikeEmbedding(
            api_base=api_base,
            api_key=api_key,
            model_name=model,
        )
        return embedding_model, model_config.name
