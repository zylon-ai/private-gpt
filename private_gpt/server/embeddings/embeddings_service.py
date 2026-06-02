from typing import Literal

from injector import inject, singleton
from pydantic import BaseModel, Field

from private_gpt.components.embedding.embedding_component import EmbeddingComponent


class Embedding(BaseModel):
    """Represents a vector embedding for a piece of text content."""

    index: int = Field(
        ...,
        description="Sequential index of this embedding in the batch, starting from 0",
        examples=[0, 1, 2],
    )
    object: Literal["embedding"] = Field(
        default="embedding",
        description="Type identifier for this object, always 'embedding'",
    )
    embedding: list[float] = Field(
        ...,
        description="High-dimensional vector representation of the text content as a list of floating-point numbers",
        examples=[[0.0023064255, -0.009327292, 0.0156234, -0.0087456]],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "index": 0,
                    "object": "embedding",
                    "embedding": [
                        0.0023064255,
                        -0.009327292,
                        0.0156234,
                        -0.0087456,
                        0.0234567,
                    ],
                },
                {
                    "index": 1,
                    "object": "embedding",
                    "embedding": [
                        -0.0045123,
                        0.0167845,
                        -0.0098234,
                        0.0134567,
                        -0.0076543,
                    ],
                },
            ]
        }
    }


@singleton
class EmbeddingsService:
    @inject
    def __init__(self, embedding_component: EmbeddingComponent) -> None:
        self.embedding_component = embedding_component

    def texts_embeddings(self, model: str, texts: list[str]) -> list[Embedding]:
        embedding_model = self.embedding_component.get_embed(model)
        texts_embeddings = embedding_model.get_text_embedding_batch(texts)
        return [
            Embedding(
                index=texts_embeddings.index(embedding),
                object="embedding",
                embedding=embedding,
            )
            for embedding in texts_embeddings
        ]
