import logging

from injector import inject, singleton
from llama_index.core.embeddings import BaseEmbedding, MockEmbedding

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
            case "huggingface":
                try:
                    from llama_index.embeddings.huggingface import (  # type: ignore
                        HuggingFaceEmbedding,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Local dependencies not found, install with `poetry install --extras embeddings-huggingface`"
                    ) from e

                self.embedding_model = HuggingFaceEmbedding(
                    model_name=settings.huggingface.embedding_hf_model_name,
                    cache_folder=str(models_cache_path),
                )
            case "sagemaker":
                try:
                    from private_gpt.components.embedding.custom.sagemaker import (
                        SagemakerEmbedding,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Sagemaker dependencies not found, install with `poetry install --extras embeddings-sagemaker`"
                    ) from e

                self.embedding_model = SagemakerEmbedding(
                    endpoint_name=settings.sagemaker.embedding_endpoint_name,
                )
            case "openai":
                try:
                    from llama_index.embeddings.openai import (  # type: ignore
                        OpenAIEmbedding,
                    )
                except ImportError as e:
                    raise ImportError(
                        "OpenAI dependencies not found, install with `poetry install --extras embeddings-openai`"
                    ) from e

                openai_settings = settings.openai.api_key
                self.embedding_model = OpenAIEmbedding(api_key=openai_settings)
            case "ollama":
                try:
                    from llama_index.embeddings.ollama import (  # type: ignore
                        OllamaEmbedding,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Local dependencies not found, install with `poetry install --extras embeddings-ollama`"
                    ) from e

                ollama_settings = settings.ollama
                self.embedding_model = OllamaEmbedding(
                    model_name=ollama_settings.embedding_model,
                    base_url=ollama_settings.embedding_api_base,
                )
            case "azopenai":
                try:
                    from llama_index.embeddings.azure_openai import (  # type: ignore
                        AzureOpenAIEmbedding,
                    )
                except ImportError as e:
                    raise ImportError(
                        "Azure OpenAI dependencies not found, install with `poetry install --extras embeddings-azopenai`"
                    ) from e

                azopenai_settings = settings.azopenai
                self.embedding_model = AzureOpenAIEmbedding(
                    model=azopenai_settings.embedding_model,
                    deployment_name=azopenai_settings.embedding_deployment_name,
                    api_key=azopenai_settings.api_key,
                    azure_endpoint=azopenai_settings.azure_endpoint,
                    api_version=azopenai_settings.api_version,
                )
            case "mock":
                # Not a random number, is the dimensionality used by
                # the default embedding model
                self.embedding_model = MockEmbedding(384)
