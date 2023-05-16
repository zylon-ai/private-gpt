from abc import ABC, abstractmethod, abstractproperty


class VectorStoreAdapter(ABC):
    @abstractproperty
    def db(self):
        pass

    @abstractmethod
    def from_documents(self, documents, embeddings, **kwargs):
        pass
