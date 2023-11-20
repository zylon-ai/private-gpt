import logging
import typing

from injector import inject, singleton
from llama_index import VectorStoreIndex
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.vector_stores.types import VectorStore

from private_gpt.components.vector_store.batched_chroma import BatchedChromaVectorStore
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.paths import local_data_path
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


@typing.no_type_check
def _chromadb_doc_id_metadata_filter(
    context_filter: ContextFilter | None,
) -> dict | None:
    if context_filter is None or context_filter.docs_ids is None:
        return {}  # No filter
    elif len(context_filter.docs_ids) < 1:
        return {"doc_id": "-"}  # Effectively filtering out all docs
    else:
        doc_filter_items = []
        if len(context_filter.docs_ids) > 1:
            doc_filter = {"$or": doc_filter_items}
            for doc_id in context_filter.docs_ids:
                doc_filter_items.append({"doc_id": doc_id})
        else:
            doc_filter = {"doc_id": context_filter.docs_ids[0]}
        return doc_filter


@singleton
class VectorStoreComponent:
    vector_store: VectorStore

    @inject
    def __init__(self, settings: Settings) -> None:
        match settings.vectorstore.database:
            case "chroma":
                try:
                    import chromadb  # type: ignore
                    from chromadb.config import (  # type: ignore
                        Settings as ChromaSettings,
                    )
                except ImportError as e:
                    raise ImportError(
                        "'chromadb' is not installed."
                        "To use PrivateGPT with Chroma, install the 'chroma' extra."
                        "`poetry install --extras chroma`"
                    ) from e

                chroma_settings = ChromaSettings(anonymized_telemetry=False)
                chroma_client = chromadb.PersistentClient(
                    path=str((local_data_path / "chroma_db").absolute()),
                    settings=chroma_settings,
                )
                chroma_collection = chroma_client.get_or_create_collection(
                    "make_this_parameterizable_per_api_call"
                )  # TODO

                self.vector_store = typing.cast(
                    VectorStore,
                    BatchedChromaVectorStore(
                        chroma_client=chroma_client, chroma_collection=chroma_collection
                    ),
                )

            case "qdrant":
                from llama_index.vector_stores.qdrant import QdrantVectorStore
                from qdrant_client import QdrantClient

                if settings.qdrant is None:
                    logger.info(
                        "Qdrant config not found. Using default settings."
                        "Trying to connect to Qdrant at localhost:6333."
                    )
                    client = QdrantClient()
                else:
                    client = QdrantClient(
                        **settings.qdrant.model_dump(exclude_none=True)
                    )
                self.vector_store = typing.cast(
                    VectorStore,
                    QdrantVectorStore(
                        client=client,
                        collection_name="make_this_parameterizable_per_api_call",
                    ),  # TODO
                )
            case _:
                # Should be unreachable
                # The settings validator should have caught this
                raise ValueError(
                    f"Vectorstore database {settings.vectorstore.database} not supported"
                )

    @staticmethod
    def get_retriever(
        index: VectorStoreIndex,
        context_filter: ContextFilter | None = None,
        similarity_top_k: int = 2,
    ) -> VectorIndexRetriever:
        # This way we support qdrant (using doc_ids) and chroma (using where clause)
        return VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            doc_ids=context_filter.docs_ids if context_filter else None,
            vector_store_kwargs={
                "where": _chromadb_doc_id_metadata_filter(context_filter)
            },
        )

    def close(self) -> None:
        if hasattr(self.vector_store.client, "close"):
            self.vector_store.client.close()
