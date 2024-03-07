import logging

from injector import inject, singleton
from llama_index.core.storage.docstore import BaseDocumentStore, SimpleDocumentStore
#from  llama_index.storage.docstore.mongodb  import MongoDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
#from llama_index.storage.index_store.redis import RedisIndexStore
from llama_index.storage.docstore.mongodb import MongoDocumentStore
from llama_index.storage.index_store.mongodb import MongoIndexStore

from llama_index.core.storage.index_store.types import BaseIndexStore

from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@singleton
class NodeStoreComponent:
    index_store: BaseIndexStore
    doc_store: MongoDocumentStore

    @inject
    def __init__(self, settings: Settings) -> None:

        try:

            if settings.indexstore.mode == "mongo":
                self.index_store = MongoIndexStore.from_host_and_port(
                        host=settings.indexstore.host,
                        port=settings.indexstore.port
                    )
            # elif settings.indexstore.mode == "redis":
            #     self.index_store = RedisIndexStore.from_host_and_port(
            #             host=settings.indexstore.host,
            #             port=settings.indexstore.port,
            #             namespace="llama"
            #         )
            else:
                self.index_store = SimpleIndexStore.from_persist_dir(
                    persist_dir=str(local_data_path)
                )
        except FileNotFoundError:
            logger.debug("Local index store not found, creating a new one")
            self.index_store = SimpleIndexStore()

        try:
            self.doc_store = MongoDocumentStore.from_host_and_port(
                host=settings.docstore.host,
                port=settings.docstore.port
            )
            #self.doc_store = SimpleDocumentStore.from_persist_dir(
            #    persist_dir=str(local_data_path)
            #)
        except FileNotFoundError:
            logger.debug("Local document store not found, creating a new one")
            self.doc_store = SimpleDocumentStore()
