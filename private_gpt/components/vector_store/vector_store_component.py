import typing

import chromadb
from injector import inject, singleton
from llama_index import VectorStoreIndex
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.vector_stores.types import VectorStore

from private_gpt.components.vector_store.batched_chroma import BatchedChromaVectorStore
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.paths import local_data_path


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
    def __init__(self) -> None:
        chroma_client = chromadb.PersistentClient(
            path=str((local_data_path / "chroma_db").absolute())
        )
        chroma_collection = chroma_client.get_or_create_collection(
            "make_this_parameterizable_per_api_call"
        )  # TODO

        self.vector_store = BatchedChromaVectorStore(
            chroma_client=chroma_client, chroma_collection=chroma_collection
        )

    @staticmethod
    def get_retriever(
        index: VectorStoreIndex,
        context_filter: ContextFilter | None = None,
        similarity_top_k: int = 2,
    ) -> VectorIndexRetriever:
        # TODO this 'where' is specific to chromadb. Implement other vector stores
        return VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            vector_store_kwargs={
                "where": _chromadb_doc_id_metadata_filter(context_filter)
            },
        )
