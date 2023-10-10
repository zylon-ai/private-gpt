from injector import inject, singleton
from llama_index.storage.docstore import BaseDocumentStore, SimpleDocumentStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.storage.index_store.types import BaseIndexStore

from private_gpt.paths import local_data_path


@singleton
class NodeStoreComponent:
    index_store: BaseIndexStore
    doc_store: BaseDocumentStore

    @inject
    def __init__(self) -> None:
        try:
            self.index_store = SimpleIndexStore.from_persist_dir(
                persist_dir=str(local_data_path)
            )
        except FileNotFoundError:
            self.index_store = SimpleIndexStore()

        try:
            self.doc_store = SimpleDocumentStore.from_persist_dir(
                persist_dir=str(local_data_path)
            )
        except FileNotFoundError:
            self.doc_store = SimpleDocumentStore()
