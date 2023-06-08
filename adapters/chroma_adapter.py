from langchain.vectorstores.chroma import Chroma

from adapters.vector_store_adapter import VectorStoreAdapter
from constants import CHROMA_SETTINGS


class ChromaVectorStoreAdapter(VectorStoreAdapter):
    def __init__(self, persist_directory, embedding_function, client_settings):
        self._persist_directory = persist_directory
        self._embedding_function = embedding_function
        self._client_settings = client_settings

        self._chroma = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_function,
            client_settings=client_settings,
        )

    @property
    def db(self):
        return self._chroma

    def from_documents(self, documents, embeddings, **kwargs):
        self._chroma.from_documents(
            documents,
            embeddings,
            persist_directory=self._persist_directory,
            client_settings=self._client_settings,
        ).persist()
