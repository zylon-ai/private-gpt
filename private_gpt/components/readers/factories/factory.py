from collections.abc import Callable

from injector import Injector, inject, singleton

from private_gpt.components.readers.factories.base import ReaderFactory
from private_gpt.components.readers.factories.docling import DoclingReaderFactory
from private_gpt.components.readers.factories.markitdown import MarkItDownReaderFactory
from private_gpt.components.readers.factories.pptx2md import PPTX2MdReaderFactory
from private_gpt.components.readers.factories.text import TextReaderFactory
from private_gpt.settings.settings import Settings

ReaderFactoryProvider = (
    type[ReaderFactory] | Callable[[Settings, Injector], ReaderFactory]
)

_PROVIDERS: dict[str, ReaderFactoryProvider] = {}


def _normalize_reader_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Reader name cannot be blank.")
    return normalized


def register_reader(name: str, provider: ReaderFactoryProvider) -> None:
    _PROVIDERS[_normalize_reader_name(name)] = provider


@singleton
class ReaderFactoryRegistry:
    @inject
    def __init__(self, settings: Settings, injector: Injector) -> None:
        built_ins: dict[str, ReaderFactory] = {
            "docling": DoclingReaderFactory(settings, injector),
            "pptx2md": PPTX2MdReaderFactory(settings, injector),
            "markitdown": MarkItDownReaderFactory(settings, injector),
            "text": TextReaderFactory(settings, injector),
        }
        self._factories: dict[str, ReaderFactory] = {
            **built_ins,
            **{name: p(settings, injector) for name, p in _PROVIDERS.items()},
        }

    def register_factory(self, name: str, factory: ReaderFactory) -> None:
        self._factories[_normalize_reader_name(name)] = factory

    def unregister_factory(self, name: str) -> None:
        self._factories.pop(_normalize_reader_name(name), None)

    def get_factory(self, name: str) -> ReaderFactory:
        normalized_name = _normalize_reader_name(name)
        factory = self._factories.get(normalized_name)
        if factory is None:
            available = ", ".join(sorted(self._factories)) or "none"
            raise ValueError(
                f"Reader '{normalized_name}' is not supported. Available: {available}"
            )
        return factory
