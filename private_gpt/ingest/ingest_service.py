from dataclasses import dataclass
from typing import BinaryIO

from injector import inject, singleton
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.vector_stores.types import VectorStore


@singleton
@inject
@dataclass
class IngestService:
    service_context: ServiceContext
    vector_store: VectorStore

    def ingest(self, file: BinaryIO) -> str:
        # load file into a LlamaIndex document
        documents = SimpleDirectoryReader(input_files=[file.name]).load_data()
        # create vectorStore index
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            service_context=self.service_context,
        )
        return file.name

    def list(self) -> set[str]:
        # TODO implement this properly, we may need a custom storage of documents
        index = VectorStoreIndex.from_vector_store(
            self.vector_store, service_context=self.service_context, show_progress=True
        )
        nodes = index.as_retriever(similarity_top_k=1000).retrieve(" ")
        files_names = {node.metadata["file_name"] for node in nodes}
        return files_names
