import logging

from injector import inject, singleton
from llama_index.core.storage.docstore import BaseDocumentStore, SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.index_store.types import BaseIndexStore

from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class NodeStoreComponent:
    index_store: BaseIndexStore
    doc_store: BaseDocumentStore

    @inject
    def __init__(self, settings: Settings) -> None:
        match settings.nodestore.database:
            case "simple":
                try:
                    self.index_store = SimpleIndexStore.from_persist_dir(
                        persist_dir=str(local_data_path)
                    )
                except FileNotFoundError:
                    logger.debug("Local index store not found, creating a new one")
                    self.index_store = SimpleIndexStore()

                try:
                    self.doc_store = SimpleDocumentStore.from_persist_dir(
                        persist_dir=str(local_data_path)
                    )
                except FileNotFoundError:
                    logger.debug("Local document store not found, creating a new one")
                    self.doc_store = SimpleDocumentStore()

            case "postgres":
                try:
                    from llama_index.storage.docstore.postgres import (  # type: ignore
                        PostgresDocumentStore,
                    )
                    from llama_index.storage.index_store.postgres import (  # type: ignore
                        PostgresIndexStore,
                    )
                except ImportError:
                    raise ImportError(
                        "Postgres dependencies not found, install with `poetry install --extras storage-nodestore-postgres`"
                    ) from None

                if settings.postgres is None:
                    raise ValueError("Postgres index/doc store settings not found.")

                self.index_store = PostgresIndexStore.from_params(
                    **settings.postgres.model_dump(exclude_none=True)
                )

                self.doc_store = PostgresDocumentStore.from_params(
                    **settings.postgres.model_dump(exclude_none=True)
                )

            case _:
                # Should be unreachable
                # The settings validator should have caught this
                raise ValueError(
                    f"Database {settings.nodestore.database} not supported"
                )
