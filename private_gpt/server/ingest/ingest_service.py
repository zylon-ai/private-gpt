from dataclasses import dataclass
from typing import BinaryIO, Dict, Any

from injector import inject, singleton
from llama_index import (
    ServiceContext,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.node_parser import SentenceWindowNodeParser
from pydantic import BaseModel, ConfigDict

from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.constants import LOCAL_DATA_PATH


@dataclass
class IngestedDoc:
    doc_id: str
    doc_metadata: Dict


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

    def ingest(self, file: BinaryIO) -> list[str]:
        # load file into a LlamaIndex document
        documents = SimpleDirectoryReader(input_files=[file.name]).load_data()
        # add doc node id to the metadata, to be able to filter during retrieval
        for document in documents:
            document.metadata["doc_id"] = document.doc_id
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
        return [document.doc_id for document in documents]

    def list_ingested(self) -> list[IngestedDoc]:
        ingested_docs = []
        try:
            docstore = self.storage_context.docstore
            ingested_docs_ids = {node.ref_doc_id for node in docstore.docs.values()}
            for doc_id in ingested_docs_ids:
                doc_metadata = docstore.get_ref_doc_info(ref_doc_id=doc_id).metadata
                # Remove unwanted info from metadata in case it exists. TODO make it a constant
                doc_metadata.pop("doc_id", None)
                doc_metadata.pop("window", None)
                doc_metadata.pop("original_text", None)
                ingested_docs.append(
                    IngestedDoc(doc_id=doc_id, doc_metadata=doc_metadata)
                )
            return ingested_docs
        except ValueError:
            pass
        return ingested_docs
