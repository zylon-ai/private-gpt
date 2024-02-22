import logging

from injector import inject, singleton
from llama_index import MockEmbedding
from llama_index.embeddings.base import BaseEmbedding

from private_gpt.paths import models_cache_path
from private_gpt.settings.settings import Settings
import torch

logger = logging.getLogger(__name__)


class CustomDataParallel(torch.nn.Module):
    def __init__(self, model):
        super(CustomDataParallel, self).__init__()
        self.model = torch.nn.DataParallel(model).cuda()

    def forward(self, **input):
        return self.model(**input)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)

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
                from transformers import AutoConfig, AutoModel

                model_name = settings.local.embedding_hf_model_name
                cache_folder = str(models_cache_path)
                # Explicitly use CUDA device 0 if available, otherwise use CPU
                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

                model = AutoModel.from_pretrained(model_name, cache_dir=cache_folder).to(device)
                tokenizer_name = model_name  # Keep tokenizer name as model name

                if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                    logger.info(f"Configuring model for multi-GPU use with {torch.cuda.device_count()} GPUs")
                    model = CustomDataParallel(model)

                self.embedding_model = HuggingFaceEmbedding(
                    model=model,
                    tokenizer_name=tokenizer_name,
                    cache_folder=cache_folder,
                    device=device,
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
