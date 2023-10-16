from injector import inject, singleton
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.embeddings.base import BaseEmbedding

from private_gpt.paths import models_cache_path
from private_gpt.settings.settings import settings


@singleton
class EmbeddingComponent:
    embedding_model: BaseEmbedding

    @inject
    def __init__(self) -> None:
        self.embedding_model = HuggingFaceEmbedding(
            model_name=settings.local.embedding_hf_model_name,
            cache_folder=str(models_cache_path),
        )
