from collections.abc import Callable, Iterator

from injector import inject, singleton
from llama_index.core.base.embeddings.base import BaseEmbedding


@singleton
class EmbeddingRegistry:
    """A registry for Embedding (Large Language Model) components with alias support.

    This class allows for the registration and retrieval of Embedding.
    """

    @inject
    def __init__(self) -> None:
        self._registry: dict[str, BaseEmbedding] = {}

    @classmethod
    def default(cls) -> str:
        """Default name for the main Embedding."""
        return "default"

    def register(
        self, name: str, component: BaseEmbedding, aliases: list[str] | None = None
    ) -> None:
        """Register a new Embedding component with a given name and optional aliases.

        :param name: The primary name of the Embedding component.
        :param component: The Embedding component to register.
        :param aliases: Optional list of aliases for this component.
        """
        aliases = [alias.strip() for alias in (aliases or []) if alias.strip()]
        all_names = [name, *aliases]

        # Check if any name is already registered
        existing_names = [n for n in all_names if n in self._registry]
        if existing_names:
            raise ValueError(f"Names already registered: {existing_names}")

        # Register component under all names
        for alias in all_names:
            self._registry[alias] = component

    def unregister(self, name: str) -> None:
        """Unregister an Embedding component by its name or alias.

        :param name: The name or alias of the Embedding component to unregister.
        """
        if name not in self._registry:
            raise KeyError(f"Embedding component '{name}' is not registered.")
        del self._registry[name]

    def get(self, name: str) -> BaseEmbedding | None:
        """Retrieve an Embedding component by its name or alias.

        :param name: The name or alias of the Embedding component to retrieve.
        :return: The registered Embedding component.
        """
        direct = self._registry.get(name)
        if direct is not None:
            return direct
        for alias, component in self._registry.items():
            if alias == name:
                return component
        return None

    def get_all(self) -> list[BaseEmbedding]:
        """Get all registered Embedding components.

        :return: List of all registered Embedding components.
        """
        return list(self._registry.values())

    def get_aliases(self, name: str) -> set[str]:
        """Get all aliases (including primary name) for a component.

        :param name: The name or alias of the component.
        :return: List of all aliases for the component.
        """
        component = self._registry.get(name)
        if component is None:
            return set[str]()

        return {alias for alias, comp in self._registry.items() if comp is component}

    def get_all_aliases(self) -> set[str]:
        """Get all registered names and aliases.

        :return: Set of all registered names and aliases.
        """
        return set(self._registry.keys())

    def filter(
        self, predicate: Callable[[BaseEmbedding], bool]
    ) -> Iterator[BaseEmbedding]:
        """Filter Embedding components based on a predicate function.

        :param predicate: A function that takes an Embedding
         and returns True if it matches the criteria.
        :return: An iterator over the filtered Embedding components.
        """
        seen = set()
        for component in self._registry.values():
            if id(component) not in seen and predicate(component):
                seen.add(id(component))
                yield component
