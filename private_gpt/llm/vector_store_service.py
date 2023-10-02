import chromadb
from injector import inject, singleton
from llama_index.vector_stores import ChromaVectorStore
from llama_index.vector_stores.types import VectorStore

from private_gpt.constants import LOCAL_DATA_PATH


@singleton
class VectorStoreService:

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
