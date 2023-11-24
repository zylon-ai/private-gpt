import logging
import multiprocessing
import os
import tempfile
from pathlib import Path
from typing import Any, BinaryIO, Literal, Sequence

from injector import inject, singleton
from llama_index import (
    Document,
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.data_structs import IndexDict
from llama_index.embeddings import BaseEmbedding
from llama_index.indices.base import BaseIndex
from llama_index.indices.utils import embed_nodes
from llama_index.ingestion import run_transformations
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.schema import BaseNode, MetadataMode
from pydantic import BaseModel, Field

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.paths import local_data_path
from private_gpt.server.ingest.ingest_helper import IngestionHelper, SimpleBulkIngestPipeline

BULK_INGEST_WORKER_NUM = max((os.cpu_count() or 1) - 1, 1)

logger = logging.getLogger(__name__)


def custom_embed_nodes(
    nodes: Sequence[BaseNode], embed_model: BaseEmbedding, show_progress: bool = False
) -> list[BaseNode]:
    id_to_embed_map = embed_nodes(
        nodes, embed_model, show_progress=show_progress
    )

    results = []
    for node in nodes:
        embedding = id_to_embed_map[node.node_id]
        result = node.copy()  # node.model_copy() does not exist
        result.embedding = embedding
        results.append(result)
    return results


class IngestedDoc(BaseModel):
    object: Literal["ingest.document"]
    doc_id: str = Field(examples=["c202d5e6-7b69-4869-81cc-dd574ee8ee11"])
    doc_metadata: dict[str, Any] | None = Field(
        examples=[
            {
                "page_label": "2",
                "file_name": "Sales Report Q3 2023.pdf",
            }
        ]
    )

    @staticmethod
    def curate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Remove unwanted metadata keys."""
        for key in ["doc_id", "window", "original_text"]:
            metadata.pop(key, None)
        return metadata


@singleton
class IngestService:
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        self.llm_service = llm_component
        self.embedding_component = embedding_component
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        self.ingest_service_context = ServiceContext.from_defaults(
            llm=self.llm_service.llm,
            embed_model=self.embedding_component.embedding_model,
            node_parser=SentenceWindowNodeParser.from_defaults(),
        )

        self._index = self._initialize_index()

        self._index_insert_lock = multiprocessing.Lock()

    def _initialize_index(self) -> BaseIndex[IndexDict]:
        """Initialize the index from the storage context."""
        try:
            # Load the index with store_nodes_override=True to be able to delete them
            index = load_index_from_storage(
                storage_context=self.storage_context,
                service_context=self.ingest_service_context,
                store_nodes_override=True,  # Force store nodes in index and document stores
                show_progress=True,
            )
        except ValueError:
            # There are no index in the storage context, creating a new one
            logger.info("Creating a new vector store index")
            index = VectorStoreIndex.from_documents(
                [],
                storage_context=self.storage_context,
                service_context=self.ingest_service_context,
                store_nodes_override=True,  # Force store nodes in index and document stores
                show_progress=True,
            )
            index.storage_context.persist(persist_dir=local_data_path)
        return index

    def _save_index(self) -> None:
        self._index.storage_context.persist(persist_dir=local_data_path)

    def ingest(self, file_name: str, file_data: Path) -> list[IngestedDoc]:
        logger.info("Ingesting file_name=%s", file_name)
        documents = IngestionHelper.transform_file_into_documents(file_name, file_data)
        logger.info(
            "Transformed file=%s into count=%s documents", file_name, len(documents)
        )
        logger.debug("Saving the documents in the index and doc store")
        return self._save_docs(documents)

    def ingest_bin_data(
        self, file_name: str, raw_file_data: BinaryIO
    ) -> list[IngestedDoc]:
        logger.debug("Ingesting binary data with file_name=%s", file_name)
        file_data = raw_file_data.read()
        logger.debug("Got file data of size=%s to ingest", len(file_data))
        # llama-index mainly supports reading from files, so
        # we have to create a tmp file to read for it to work
        # delete=False to avoid a Windows 11 permission error.
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                path_to_tmp = Path(tmp.name)
                if isinstance(file_data, bytes):
                    path_to_tmp.write_bytes(file_data)
                else:
                    path_to_tmp.write_text(str(file_data))
                return self.ingest(file_name, path_to_tmp)
            finally:
                tmp.close()
                path_to_tmp.unlink()

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> None:
        logger.info("Ingesting file_names=%s", [f[0] for f in files])
        pipeline = SimpleBulkIngestPipeline()
        pipeline.bulk_ingest(files_to_process=files, to_db_func=self._save_docs)

    def _save_docs(self, documents: list[Document]) -> list[IngestedDoc]:
        for document in documents:
            document.metadata["doc_id"] = document.doc_id
            # We don't want the Embeddings search to receive this metadata
            document.excluded_embed_metadata_keys = ["doc_id"]
            # We don't want the LLM to receive these metadata in the context
            document.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]
        logger.debug("Inserting the documents in the index")
        with self._index_insert_lock:
            # Doing the modification and saving of the index using a lock
            # to prevent concurrent writes
            # for doc in documents:
            #     self._index.insert(doc)
            self._index_insert_documents(documents)
            logger.debug("Storing the documents in the doc store")
            # persist the index and nodes
            self._save_index()
        return [
            IngestedDoc(
                object="ingest.document",
                doc_id=document.doc_id,
                doc_metadata=IngestedDoc.curate_metadata(document.metadata),
            )
            for document in documents
        ]

    def _index_insert_documents(self, documents: list[Document]) -> None:
        logger.debug("Transforming count=%s documents into nodes", len(documents))
        nodes = run_transformations(
            documents,
            self._index._service_context.transformations,
            show_progress=self._index._show_progress,
        )
        logger.debug("Embeddings count=%s nodes", len(nodes))
        nodes = custom_embed_nodes(nodes, self.embedding_component.embedding_model, show_progress=True)
        logger.debug("Inserting count=%s nodes in the index", len(nodes))
        self._index.insert_nodes(nodes, show_progress=True)
        for document in documents:
            logger.debug("Setting the document hash in the doc store for document=%s", document.doc_id)
            self._index.docstore.set_document_hash(document.get_doc_id(), document.hash)

    def list_ingested(self) -> list[IngestedDoc]:
        ingested_docs = []
        try:
            docstore = self.storage_context.docstore
            ingested_docs_ids: set[str] = set()

            for node in docstore.docs.values():
                if node.ref_doc_id is not None:
                    ingested_docs_ids.add(node.ref_doc_id)

            for doc_id in ingested_docs_ids:
                ref_doc_info = docstore.get_ref_doc_info(ref_doc_id=doc_id)
                doc_metadata = None
                if ref_doc_info is not None and ref_doc_info.metadata is not None:
                    doc_metadata = IngestedDoc.curate_metadata(ref_doc_info.metadata)
                ingested_docs.append(
                    IngestedDoc(
                        object="ingest.document",
                        doc_id=doc_id,
                        doc_metadata=doc_metadata,
                    )
                )
        except ValueError:
            logger.warning("Got an exception when getting list of docs", exc_info=True)
            pass
        logger.debug("Found count=%s ingested documents", len(ingested_docs))
        return ingested_docs

    def delete(self, doc_id: str) -> None:
        """Delete an ingested document.

        :raises ValueError: if the document does not exist
        """
        logger.info(
            "Deleting the ingested document=%s in the doc and index store", doc_id
        )

        with self._index_insert_lock:
            # Delete the document from the index
            self._index.delete_ref_doc(doc_id, delete_from_docstore=True)

            # Save the index
            self._save_index()
