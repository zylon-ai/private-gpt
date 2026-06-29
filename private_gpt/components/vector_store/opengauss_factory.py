import logging
import threading
from typing import Any

from llama_index.core.vector_stores.types import BasePydanticVectorStore

from private_gpt.components.vector_store.factory import VectorStoreFactory
from private_gpt.settings.settings import Settings
from private_gpt.utils.dependencies import format_missing_dependency_message

logger = logging.getLogger(__name__)

DEFAULT_TABLE = "embeddings"


class OpenGaussVectorStoreFactory(VectorStoreFactory):
    """Factory that builds OpenGaussVectorStore instances.

    A single psycopg2 connection is shared across all collections (guarded by
    a lock inside the store). Logical multitenancy is implemented by storing
    the collection name as a metadata field and filtering on it.
    """

    _conn: Any | None = None
    _conn_lock = threading.Lock()

    def __init__(self, settings: Settings, embed_dim: int | None = None) -> None:
        super().__init__(settings)
        self._embed_dim = embed_dim

    def _build_connection(self) -> Any:
        try:
            import psycopg2  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                format_missing_dependency_message(
                    "openGauss DataVec vector store",
                    extras="vectorstore-opengauss",
                )
            ) from e

        cfg = self.settings.opengauss
        assert cfg is not None, "openGauss settings must be provided"

        conn = psycopg2.connect(
            host=cfg.host,
            port=cfg.port,
            dbname=cfg.database,
            user=cfg.user,
            password=cfg.password,
            application_name="private-gpt",
        )
        conn.autocommit = False
        return conn

    def _ensure_connection(self) -> Any:
        if self._conn is None or self._conn.closed:
            with self._conn_lock:
                if self._conn is None or self._conn.closed:
                    self._conn = self._build_connection()
        return self._conn

    def vector_store(self, collection: str) -> BasePydanticVectorStore:
        from private_gpt.components.vector_store.opengauss_store import (
            OpenGaussVectorStore,
        )

        cfg = self.settings.opengauss
        assert cfg is not None, "openGauss settings must be provided"

        conn = self._ensure_connection()

        logical_multitenancy = self.settings.vectorstore.multitenancy == "logical"
        table_name = (
            self.settings.vectorstore.default_collection or DEFAULT_TABLE
            if logical_multitenancy
            else collection
        )

        store = OpenGaussVectorStore(
            connection=conn,
            schema_name=cfg.schema_name,
            table_name=table_name,
            embed_dim=self._embed_dim or self.settings.vectorstore.embed_dim,
            distance=cfg.distance,
        )

        if logical_multitenancy:
            logger.info(
                "openGauss logical multitenancy: all collections share table %s, "
                "filtered by '%s' metadata",
                table_name,
                "collection",
            )

        return store

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None
