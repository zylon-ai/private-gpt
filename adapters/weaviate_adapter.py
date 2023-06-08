import weaviate
from langchain.vectorstores.weaviate import Weaviate

from adapters.vector_store_adapter import VectorStoreAdapter


class WeaviateVectorStoreAdapter(VectorStoreAdapter):
    def __init__(
        self,
        weaviate_url="http://localhost:8080",
        index_name="Langchain",
        text_key="text",
        embedding=None,
    ):
        self._url = weaviate_url
        self._index_name = index_name
        self._text_key = text_key

        client = weaviate.Client(weaviate_url)
        self._weaviate = Weaviate(
            client,
            index_name,
            text_key,
            by_text=False,
            embedding=embedding,
            attributes=["source"],
        )

    @property
    def db(self):
        return self._weaviate

    def from_documents(self, documents, embeddings, **kwargs):
        self._weaviate.from_documents(
            documents,
            embeddings,
            weaviate_url=self._url,
            index_name=self._index_name,
            text_key=self._text_key,
        )
