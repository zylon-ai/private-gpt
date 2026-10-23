from llama_index.core import MockEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding

from private_gpt.components.embedding.factories.base import EmbeddingFactory
from private_gpt.settings.settings import EmbeddingModelConfig, Settings


class MockEmbeddingFactory(EmbeddingFactory):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)

    def _create_embedding(
        self, model_config: EmbeddingModelConfig
    ) -> tuple[BaseEmbedding, str | None]:
        return MockEmbedding(model_config.context_window), model_config.name
