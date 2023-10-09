import typing

import chromadb
from injector import inject, singleton
from llama_index import VectorStoreIndex
from llama_index.indices.vector_store import VectorIndexRetriever
from llama_index.vector_stores import ChromaVectorStore
from llama_index.vector_stores.types import VectorStore

from private_gpt.constants import LOCAL_DATA_PATH
from private_gpt.open_ai.extensions.context_docs import ContextDocs


@typing.no_type_check
def _chromadb_doc_id_metadata_filter(context_docs: ContextDocs) -> dict | None:
    if context_docs.docs_ids is None or len(context_docs.docs_ids) < 1:
        return {"doc_id": "-"}
    elif context_docs.docs_ids == "all":
        return None  # No filtering
    else:
        doc_filter_items = []
        if len(context_docs.docs_ids) > 1:
            doc_filter = {"$or": doc_filter_items}
            for doc_id in context_docs.docs_ids:
                doc_filter_items.append({"doc_id": doc_id})
        else:
            doc_filter = {"doc_id": context_docs.docs_ids[0]}
        return doc_filter


@singleton
class VectorStoreComponent:
    vector_store: VectorStore

    @inject
    def __init__(self) -> None:
        db = chromadb.PersistentClient(
            path=str((LOCAL_DATA_PATH / "chroma_db").absolute())
        )
        chroma_collection = db.get_or_create_collection(
            "make_this_parameterizable_per_api_call"
        )  # TODO

        self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    @staticmethod
    def get_retriever(
        index: VectorStoreIndex, context_docs: ContextDocs, similarity_top_k: int = 2
    ) -> VectorIndexRetriever:
        # TODO this 'where' is specific to chromadb. Implement other vector stores
        return VectorIndexRetriever(
            index=index,
            similarity_top_k=similarity_top_k,
            vector_store_kwargs={
                "where": _chromadb_doc_id_metadata_filter(context_docs)
            },
        )
