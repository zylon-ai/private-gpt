from injector import inject, singleton
from llama_index.embeddings import resolve_embed_model
from llama_index.embeddings.base import BaseEmbedding


@singleton
class EmbeddingComponent:
    embedding_model: BaseEmbedding

    @inject
    def __init__(self) -> None:
        self.embedding_model = resolve_embed_model("local")
