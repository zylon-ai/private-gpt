from abc import ABC, abstractmethod
from collections.abc import Callable

from llama_index.core.vector_stores.types import BasePydanticVectorStore

from private_gpt.settings.settings import Settings

VectorStoreProvider = (
    type["VectorStoreFactory"] | Callable[[Settings, int | None], "VectorStoreFactory"]
)

_PROVIDERS: dict[str, VectorStoreProvider] = {}


def register_vector_store(database: str, provider: VectorStoreProvider) -> None:
    _PROVIDERS[database] = provider


class VectorStoreFactory(ABC):
    def __init__(self, settings: Settings, embed_dim: int | None = None) -> None:
        self.settings = settings
        self.embed_dim = embed_dim

    def warm_up(self) -> None:
        return None

    @abstractmethod
    def vector_store(self, collection: str) -> BasePydanticVectorStore: ...
