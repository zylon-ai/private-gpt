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
        return EmbeddingInstance(embedding=embedding, alias=alias)

    @abstractmethod
    def _create_embedding(
        self, model_config: EmbeddingModelConfig
    ) -> tuple[BaseEmbedding, str | None]:
        """Create Embedding instance, to be implemented by subclasses."""
        pass
