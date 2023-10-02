from dataclasses import dataclass
from typing import BinaryIO

from injector import inject, singleton
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)

from private_gpt.llm.llm_service import LLMService
from private_gpt.llm.vector_store_service import VectorStoreService


@singleton
@dataclass
class IngestService:
    @inject
    def __init__(
        self, llm_service: LLMService, vector_store_service: VectorStoreService
    ) -> None:
        self.llm_service = llm_service
        self.vector_store_service = vector_store_service
        self.storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store_service.vector_store
        )
        self.ingest_service_context = ServiceContext.from_defaults(
            llm=self.llm_service.llm, embed_model="local"
        )

    def ingest(self, file: BinaryIO) -> str:
        # load file into a LlamaIndex document
        documents = SimpleDirectoryReader(input_files=[file.name]).load_data()
        # create vectorStore index

        VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            service_context=self.ingest_service_context,
        )
        return file.name

    def list(self) -> set[str]:
        # TODO implement this properly, we may need a custom storage of documents
        index = VectorStoreIndex.from_vector_store(
            self.vector_store_service.vector_store,
            service_context=self.ingest_service_context,
            show_progress=True,
        )
        nodes = index.as_retriever(similarity_top_k=1000).retrieve(" ")
        files_names = {node.metadata["file_name"] for node in nodes}
        return files_names
