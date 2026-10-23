from collections.abc import Callable, Iterator

from injector import inject, singleton
from llama_index.core.llms import LLM
from pydantic import BaseModel

from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase


class LLMInstance(BaseModel):
    llm: LLM
    tokenizer: TokenizerBase | None = None

    class Config:
        arbitrary_types_allowed = True


@singleton
class LLMRegistry:
    """A registry for LLM (Large Language Model) components with alias support.

    This class allows for the registration and retrieval of LLM.
    """

    @inject
    def __init__(self) -> None:
        self._registry: dict[str, LLMInstance] = {}

    @classmethod
    def default(cls) -> str:
        """Default name for the main LLM."""
        return "default"

    def register(
        self, name: str, component: LLMInstance, aliases: list[str] | None = None
    ) -> None:
        """Register a new LLM component with a given name and optional aliases.

        :param name: The primary name of the LLM component.
        :param component: The LLM component to register.
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
        """Unregister an LLM component by its name or alias.

        :param name: The name or alias of the LLM component to unregister.
        """
        if name not in self._registry:
            raise KeyError(f"LLM component '{name}' is not registered.")
        del self._registry[name]

    def get(self, name: str) -> LLMInstance | None:
        """Retrieve an LLM component by its name or alias.

        :param name: The name or alias of the LLM component to retrieve.
        :return: The registered LLM component.
        """
        direct = self._registry.get(name)
        if direct is not None:
            return direct
        for alias, component in self._registry.items():
            if alias == name:
                return component
        return None

    def get_all(self) -> list[LLMInstance]:
        """Get all registered LLM components.

        :return: List of all registered LLM components.
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

    def filter(self, predicate: Callable[[LLMInstance], bool]) -> Iterator[LLMInstance]:
        """Filter LLM components based on a predicate function.

        :param predicate: A function that takes an LLM
         and returns True if it matches the criteria.
        :return: An iterator over the filtered LLM components.
        """
        seen = set()
        for component in self._registry.values():
            if id(component) not in seen and predicate(component):
                seen.add(id(component))
                yield component
