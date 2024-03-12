import logging

from injector import inject, singleton
from private_gpt.settings.settings import Settings
from llama_index.core.storage.docstore import BaseDocumentStore, SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.index_store.types import BaseIndexStore

from private_gpt.paths import local_data_path

logger = logging.getLogger(__name__)


@singleton
class NodeStoreComponent:
    index_store: BaseIndexStore
    doc_store: BaseDocumentStore

    @inject
    def __init__(self, settings: Settings) -> None:
        match settings.docstore.database:
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
                    from llama_index.core.storage.index_store.postgres_index_store import PostgresIndexStore
                    from llama_index.core.storage.docstore.postgres_docstore import PostgresDocumentStore
                except Import Error as e:
                    raise ImportError (
                        "Postgres dependencies not found, install with `poetry install --extras storage-postgres`"
                        )
                
                if settings.postgres is None:
                    raise ValueError("Postgres index/doc store settings not found.")

                self.index_store = PostgresIndexStore.from_params(**settings.postgres.model_dump(exclude_none=True))
                self.doc_store = PostgresDocumentStore.from_params(**settings.postgres.model_dump(exclude_none=True))

            case _:
                # Should be unreachable
                # The settings validator should have caught this
                raise ValueError(
                    f"Database {settings.docstore.database} not supported"
                )
