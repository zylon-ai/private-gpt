from dataclasses import dataclass
from typing import BinaryIO

from injector import inject, singleton
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.node_parser import SentenceWindowNodeParser

from private_gpt.llm.llm_service import LLMService
from private_gpt.node_store.node_store_service import NodeStoreService
from private_gpt.vector_store.vector_store_service import VectorStoreService


@singleton
@dataclass
class IngestService:
    @inject
    def __init__(
        self,
        llm_service: LLMService,
        vector_store_service: VectorStoreService,
        node_store_service: NodeStoreService,
    ) -> None:
        self.llm_service = llm_service
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_service.vector_store,
            docstore=node_store_service.doc_store,
            index_store=node_store_service.index_store,
        )
        self.ingest_service_context = ServiceContext.from_defaults(
            llm=self.llm_service.llm,
            embed_model="local",
            node_parser=SentenceWindowNodeParser.from_defaults(),
        )

    def ingest(self, file: BinaryIO) -> str:
        # load file into a LlamaIndex document
        documents = SimpleDirectoryReader(input_files=[file.name]).load_data()
        # create vectorStore index
        VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            service_context=self.ingest_service_context,
            store_nodes_override=True,  # Force store nodes in index store and document store
            show_progress=True,
        )
        # persist the index and nodes
        self.storage_context.persist()
        return file.name

    def list(self) -> set[str]:
        file_names = []
        try:
            docstore = self.storage_context.docstore
            file_names = {
                ref_doc.metadata["file_name"] for ref_doc in docstore.docs.values()
            }
        except ValueError:
            pass
        return file_names
