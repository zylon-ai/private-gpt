from injector import inject, singleton

from private_gpt.components.embedding.embedding_component import EmbeddingComponent


@singleton
class EmbeddingsService:
    @inject
    def __init__(self, embedding_component: EmbeddingComponent) -> None:
        self.embedding_model = embedding_component.embedding_model

    def texts_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self.embedding_model.get_text_embedding_batch(texts)
