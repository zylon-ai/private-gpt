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
        embedding_mode = settings.embedding.mode
        logger.info("Initializing the embedding model in mode=%s", embedding_mode)
        match embedding_mode:
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

            case "bedrock":

                from llama_index.embeddings import BedrockEmbedding

                self.embedding_model = BedrockEmbedding(
                    model_name=settings.bedrock.embedding_modelid,
                )

                from boto3 import Session

                # Access credentials using boto3
                session = Session()
                credentials = session.get_credentials()

                # Access key ID and secret access key
                access_key = credentials.access_key
                secret_key = credentials.secret_key

                self.embedding_model.set_credentials(aws_region=settings.bedrock.region,
                                                     aws_access_key_id=access_key,
                                                     aws_secret_access_key=secret_key
                                                     )