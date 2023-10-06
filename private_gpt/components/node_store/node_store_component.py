from injector import inject, singleton
from llama_index.storage.docstore import BaseDocumentStore, SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.storage.index_store.types import BaseIndexStore

from private_gpt.constants import LOCAL_DATA_PATH


@singleton
class NodeStoreComponent:
    index_store: BaseIndexStore
    doc_store: BaseDocumentStore

    @inject
    def __init__(self) -> None:
        try:
            self.index_store = SimpleIndexStore.from_persist_dir(
                persist_dir=str(LOCAL_DATA_PATH)
            )
        except FileNotFoundError:
            self.index_store = SimpleIndexStore()

        try:
            self.doc_store = SimpleDocumentStore.from_persist_dir(
                persist_dir=str(LOCAL_DATA_PATH)
            )
        except FileNotFoundError:
            self.doc_store = SimpleDocumentStore()
