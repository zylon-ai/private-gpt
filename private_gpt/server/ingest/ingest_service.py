import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, AnyStr, Literal

from injector import inject, singleton
from llama_index import (
    Document,
    JSONReader,
    ServiceContext,
    StorageContext,
    StringIterableReader,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.readers.file.base import DEFAULT_FILE_READER_CLS
from pydantic import BaseModel, Field

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.paths import local_data_path

if TYPE_CHECKING:
    from llama_index.readers.base import BaseReader

# Patching the default file reader to support other file types
FILE_READER_CLS = DEFAULT_FILE_READER_CLS.copy()
FILE_READER_CLS.update(
    {
        ".json": JSONReader,
    }
)

logger = logging.getLogger(__name__)


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
        metadata.pop("doc_id", None)
        metadata.pop("window", None)
        metadata.pop("original_text", None)
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
        self.storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        self.ingest_service_context = ServiceContext.from_defaults(
            llm=self.llm_service.llm,
            embed_model=embedding_component.embedding_model,
            node_parser=SentenceWindowNodeParser.from_defaults(),
        )

    def ingest(self, file_name: str, file_data: AnyStr | Path) -> list[IngestedDoc]:
        logger.info("Ingesting file_name=%s", file_name)
        extension = Path(file_name).suffix
        reader_cls = FILE_READER_CLS.get(extension)
        documents: list[Document]
        if reader_cls is None:
            logger.debug(
                "No reader found for extension=%s, using default string reader",
                extension,
            )
            # Read as a plain text
            string_reader = StringIterableReader()
            if isinstance(file_data, Path):
                text = file_data.read_text()
                documents = string_reader.load_data([text])
            elif isinstance(file_data, bytes):
                documents = string_reader.load_data([file_data.decode("utf-8")])
            elif isinstance(file_data, str):
                documents = string_reader.load_data([file_data])
            else:
                raise ValueError(f"Unsupported data type {type(file_data)}")
        else:
            logger.debug("Specific reader found for extension=%s", extension)
            reader: BaseReader = reader_cls()
            if isinstance(file_data, Path):
                # Already a path, nothing to do
                documents = reader.load_data(file_data)
            else:
                # llama-index mainly supports reading from files, so
                # we have to create a tmp file to read for it to work
                with tempfile.NamedTemporaryFile() as tmp:
                    path_to_tmp = Path(tmp.name)
                    if isinstance(file_data, bytes):
                        path_to_tmp.write_bytes(file_data)
                    else:
                        path_to_tmp.write_text(str(file_data))
                    documents = reader.load_data(path_to_tmp)
        logger.info(
            "Transformed file=%s into count=%s documents", file_name, len(documents)
        )
        for document in documents:
            document.metadata["file_name"] = file_name
        return self._save_docs(documents)

    def _save_docs(self, documents: list[Document]) -> list[IngestedDoc]:
        for document in documents:
            document.metadata["doc_id"] = document.doc_id
            # We don't want the Embeddings search to receive this metadata
            document.excluded_embed_metadata_keys = ["doc_id"]
            # We don't want the LLM to receive these metadata in the context
            document.excluded_llm_metadata_keys = ["file_name", "doc_id", "page_label"]
        # create vectorStore index
        VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            service_context=self.ingest_service_context,
            store_nodes_override=True,  # Force store nodes in index and document stores
            show_progress=True,
        )
        # persist the index and nodes
        self.storage_context.persist(persist_dir=local_data_path)
        return [
            IngestedDoc(
                object="ingest.document",
                doc_id=document.doc_id,
                doc_metadata=IngestedDoc.curate_metadata(document.metadata),
            )
            for document in documents
        ]

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

        # Load the index with store_nodes_override=True to be able to delete them
        index = load_index_from_storage(self.storage_context, store_nodes_override=True)

        # Delete the document from the index
        index.delete_ref_doc(doc_id, delete_from_docstore=True)

        # Save the index
        self.storage_context.persist(persist_dir=local_data_path)
