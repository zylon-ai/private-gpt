from collections.abc import Callable

from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.readers.base_reader import IngestionReader
from private_gpt.components.readers.factories.base import ReaderFactory

DoclingModeProvider = Callable[["DoclingReaderFactory"], IngestionReader]


def _create_docling_api_reader(factory: "DoclingReaderFactory") -> IngestionReader:
    from private_gpt.components.readers.docling.docling_api_reader import (
        DoclingApiReader,
    )

    llm_component = factory.injector.get(LLMComponent)
    return DoclingApiReader(
        factory.settings.docling,
        reader_settings=factory.settings.transformation.docling,
        llm_component=llm_component,
    )


_PROVIDERS: dict[str, DoclingModeProvider] = {
    "api": _create_docling_api_reader,
}


def register_docling_mode(mode: str, provider: DoclingModeProvider) -> None:
    _PROVIDERS[mode] = provider


class DoclingReaderFactory(ReaderFactory):
    def create_reader(self, extension: str | None = None) -> IngestionReader:
        del extension

        docling_settings = self.settings.docling
        provider = _PROVIDERS.get(docling_settings.mode)
        if provider is None:
            raise ValueError(f"Unknown Docling mode: {docling_settings.mode}")
        return provider(self)
