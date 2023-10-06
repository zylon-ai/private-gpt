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

from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.constants import LOCAL_DATA_PATH


@singleton
class IngestService:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.llm_service = llm_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
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
        self.storage_context.persist(persist_dir=LOCAL_DATA_PATH)
        return file.name

    def list(self) -> set[str]:
        try:
            docstore = self.storage_context.docstore
            file_names = {
                ref_doc.metadata["file_name"] for ref_doc in docstore.docs.values()
            }
        except ValueError:
            file_names = set()
        return file_names
