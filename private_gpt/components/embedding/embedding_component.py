import logging

from injector import inject, singleton
from llama_index import MockEmbedding
from llama_index.embeddings.base import BaseEmbedding

from private_gpt.paths import models_cache_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class EmbeddingComponent:
    embedding_model: BaseEmbedding

    @inject
    def __init__(self, settings: Settings) -> None:
        embedding_mode = settings.chatbot.embedding_mode  
        chatbot_mode = settings.chatbot.mode

        if chatbot_mode == "hybrid":
            embedding_mode = settings.chatbot.embedding_mode

        match embedding_mode:
            case "local":
                self.embedding_model = self._create_local_embedding(settings)
            case "sagemaker":
                self.embedding_model = self._create_sagemaker_embedding(settings)
            case "openai":
                self.embedding_model = self._create_openai_embedding(settings)
            case "mock":
                self.embedding_model = MockEmbedding(384)
            case _:
                raise ValueError(f"Unsupported embedding mode chatbot: {chatbot_mode}") 

    def _create_local_embedding(self, settings: Settings) -> BaseEmbedding:
        from llama_index.embeddings import HuggingFaceEmbedding
        return HuggingFaceEmbedding(
            model_name=settings.local.embedding_hf_model_name,
            cache_folder=str(models_cache_path),
        )

    def _create_sagemaker_embedding(self, settings: Settings) -> BaseEmbedding:
        from private_gpt.components.embedding.custom.sagemaker import SagemakerEmbedding
        return SagemakerEmbedding(
            endpoint_name=settings.sagemaker.embedding_endpoint_name,
        )

    def _create_openai_embedding(self, settings: Settings) -> BaseEmbedding:
        from llama_index import OpenAIEmbedding
        openai_settings = settings.openai.api_key
        return OpenAIEmbedding(api_key=openai_settings)

