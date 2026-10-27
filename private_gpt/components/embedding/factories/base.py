import logging
from abc import ABC, abstractmethod

from llama_index.core.base.embeddings.base import BaseEmbedding
from pydantic import BaseModel, Field

from private_gpt.settings.settings import EmbeddingModelConfig, Settings

logger = logging.getLogger(__name__)


class EmbeddingInstance(BaseModel):
    embedding: BaseEmbedding = Field(..., description="The Embedding instance")
    alias: str | None = Field(None, description="Optional alias for the Embedding")

    class Config:
        arbitrary_types_allowed = True


class EmbeddingFactory(ABC):
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def create_embedding(self, model_config: EmbeddingModelConfig) -> EmbeddingInstance:
        """Create Embedding instance and return the instance and config."""
        embedding, alias = self._create_embedding(model_config)

        # llama_index batches the texts of one embedding request by
        # ``embed_batch_size`` (see ``BaseEmbedding.get_text_embedding_batch``),
        # so the per-model setting has to reach the instance. Factories do not
        # know about it, and leaving the library default (100 texts for OpenAI,
        # 10 for the rest) ignores ``embedding_batch_size`` and can exceed the
        # batch limit of a self-hosted embedding server. Non-positive values
        # keep the library default, as with other numeric settings.
        batch_size = model_config.embedding_batch_size
        if batch_size is not None and batch_size > 0:
            embedding.embed_batch_size = batch_size

        return EmbeddingInstance(embedding=embedding, alias=alias)

    @abstractmethod
    def _create_embedding(
        self, model_config: EmbeddingModelConfig
    ) -> tuple[BaseEmbedding, str | None]:
        """Create Embedding instance, to be implemented by subclasses."""
        pass
