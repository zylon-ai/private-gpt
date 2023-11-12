from injector import inject, singleton
from llama_index import MockEmbedding
from llama_index.embeddings.base import BaseEmbedding

from private_gpt.paths import models_cache_path
from private_gpt.settings.settings import Settings


@singleton
class EmbeddingComponent:
    embedding_model: BaseEmbedding

    @inject
    def __init__(self, settings: Settings) -> None:
        match settings.llm.mode:
            case "local":
                from llama_index.embeddings import HuggingFaceEmbedding

                self.embedding_model = HuggingFaceEmbedding(
                    model_name=settings.local.embedding_hf_model_name,
                    cache_folder=str(models_cache_path),
                )
            case "sagemaker":

                from private_gpt.components.embedding.custom.sagemaker import (
                    SagemakerEmbedding,
                )

                self.embedding_model = SagemakerEmbedding(
                    endpoint_name=settings.sagemaker.embedding_endpoint_name,
                )
            case "openai":
                from llama_index import OpenAIEmbedding

                openai_settings = settings.openai.api_key
                self.embedding_model = OpenAIEmbedding(api_key=openai_settings)
            case "mock":
                # Not a random number, is the dimensionality used by
                # the default embedding model
                self.embedding_model = MockEmbedding(384)
