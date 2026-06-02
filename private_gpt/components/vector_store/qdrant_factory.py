import typing
from typing import Any

from llama_index.core.vector_stores.types import BasePydanticVectorStore

from private_gpt.components.ingest.metadata_helper import MetadataKeys
from private_gpt.components.vector_store.factory import VectorStoreFactory
from private_gpt.settings.settings import Settings
from private_gpt.utils.dependencies import format_missing_dependency_message

DEFAULT_COLLECTION = "zgptvector"


class QdrantVectorStoreFactory(VectorStoreFactory):
    def __init__(self, settings: Settings, embed_dim: int | None = None) -> None:
        super().__init__(settings)
        self._client: Any | None = None
        self._embed_dim = embed_dim

    def _build_client(self) -> Any:
        try:
            from private_gpt.components.vector_store.qdrant_client_builder import (
                QdrantClientBuilder,
            )
        except ImportError as e:
            raise ImportError(
                format_missing_dependency_message(
                    "Qdrant vector store",
                    extras="vectorstore-qdrant",
                )
            ) from e

        return QdrantClientBuilder.build_clients(self.settings)

    def _ensure_client(self) -> None:
        if self._client is None:
            self._client = self._build_client()

    def vector_store(self, collection: str) -> BasePydanticVectorStore:
        try:
            from qdrant_client import models  # type: ignore

            from private_gpt.components.vector_store.patched_qdrant_store import (
                PatchedQdrantVectorStore,
            )
        except ImportError as e:
            raise ImportError(
                format_missing_dependency_message(
                    "Qdrant vector store",
                    extras="vectorstore-qdrant",
                )
            ) from e

        logical_multitenancy = self.settings.vectorstore.multitenancy == "logical"
        default_collection = (
            self.settings.vectorstore.default_collection or DEFAULT_COLLECTION
        )

        self._ensure_client()
        assert (
            self._client is not None
        ), "Qdrant client should be initialized at this point"

        return typing.cast(
            BasePydanticVectorStore,
            PatchedQdrantVectorStore(
                client=self._client.client,
                aclient=self._client.aclient,
                collection_name=(
                    collection if not logical_multitenancy else default_collection
                ),
                group_id=collection if logical_multitenancy else None,
                group_id_field=MetadataKeys.COLLECTION.value,
                enable_hybrid=self.settings.qdrant.hybrid_search,
                embed_dim=self._embed_dim or self.settings.vectorstore.embed_dim,
                distance=self.settings.qdrant.distance_metric,
                logical_multitenancy=logical_multitenancy,
                hnsw_m=self.settings.qdrant.hnsw_m,
                hnsw_payload_m=self.settings.qdrant.hnsw_payload_m,
                indexes={
                    MetadataKeys.ARTIFACT_ID.value: models.KeywordIndexType.KEYWORD,
                    MetadataKeys.PROJECT_ID.value: models.KeywordIndexType.KEYWORD,
                },
            ),
        )

    def close(self) -> None:
        if self._client is not None and hasattr(self._client, "close"):
            self._client.close()
        self._client = None
