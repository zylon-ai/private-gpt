import logging

from injector import inject, singleton
from llama_index.core.embeddings import BaseEmbedding, MockEmbedding

from private_gpt.paths import models_cache_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)
import torch

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

                # Get the number of available GPUs
                num_gpus = torch.cuda.device_count()

                if num_gpus > 0:
                    print("Available CUDA devices:")
                    for i in range(num_gpus):
                        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                else:
                    print("No CUDA devices available. Switching to CPU.")

                # Check if CUDA is available
                if torch.cuda.is_available():
                    # If settings.embedding.gpu is specified, use that GPU index
                    if hasattr(settings, 'huggingface') and hasattr(settings.huggingface, 'gpu_type'):
                        device = torch.device(f"{settings.huggingface.gpu_type}:{settings.huggingface.gpu_number}")
                    else:
                        device = torch.device('cuda:0')
                else:
                    # If CUDA is not available, use CPU
                    device = torch.device("cpu")
                print("Embedding Device: ",device)
                self.embedding_model = HuggingFaceEmbedding(
                    model_name=settings.huggingface.embedding_hf_model_name,
                    cache_folder=str(models_cache_path),
                    device=device
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
                    base_url=ollama_settings.api_base,
                )
            case "mock":
                # Not a random number, is the dimensionality used by
                # the default embedding model
                self.embedding_model = MockEmbedding(384)
