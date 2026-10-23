from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from llama_index.core.schema import BaseNode, Document, MetadataMode

from private_gpt.components.ingest.parse_component import ParseComponent
from private_gpt.components.ingest.utils import get_file_info
from private_gpt.components.readers.base_reader import IngestionReader
from private_gpt.components.readers.factories.base import ReaderFactory
from private_gpt.components.readers.pdf_inspector.pdf_inspector_reader import (
    PdfInspectorFallbackError,
)
from private_gpt.components.readers.reader_component import ReaderComponent
from private_gpt.settings.settings import Settings
from tests.fixtures.mock_injector import MockInjector

TEST_FOLDER_PATH = Path(__file__).parents[0]


class _FallbackReader(IngestionReader):
    async def lazy_load_data(
        self,
        file_info: Any,
        extra_info: dict[str, Any] | None = None,
        execute_transformations: bool = True,
        *args: Any,
        **load_kwargs: Any,
    ) -> AsyncIterator[BaseNode]:
        del file_info, execute_transformations, args, load_kwargs
        if False:
            yield Document(text="")
        raise PdfInspectorFallbackError("needs OCR")


class _FallbackReaderFactory(ReaderFactory):
    def create_reader(self, extension: str | None = None) -> IngestionReader:
        del extension
        return _FallbackReader()


class _StubDoclingReader(IngestionReader):
    async def lazy_load_data(
        self,
        file_info: Any,
        extra_info: dict[str, Any] | None = None,
        execute_transformations: bool = True,
        *args: Any,
        **load_kwargs: Any,
    ) -> AsyncIterator[BaseNode]:
        del file_info, execute_transformations, args, load_kwargs
        yield Document(
            text="docling content",
            extra_info=extra_info if extra_info is not None else {},
        )


class _StubDoclingReaderFactory(ReaderFactory):
    def create_reader(self, extension: str | None = None) -> IngestionReader:
        del extension
        return _StubDoclingReader()


def test_pdf_inspector_fallback_uses_next_reader_in_chain(
    injector: MockInjector,
) -> None:
    injector.bind_settings(
        {
            "data": {
                "reader": "auto",
                "local_data_folder": "local_data/tests",
                "enable_vision_fallback": False,
            }
        }
    )
    reader_component = injector.get(ReaderComponent)
    original_reader_names = reader_component.get_reader_names(extension=".pdf")

    reader_component.register_reader_factory(
        "pdf-inspector-fallback-stub",
        _FallbackReaderFactory(injector.get(Settings), injector.test_injector),
    )
    reader_component.register_reader_factory(
        "docling-stub",
        _StubDoclingReaderFactory(injector.get(Settings), injector.test_injector),
    )
    try:
        reader_component.register_extension_readers(
            ".pdf",
            ["pdf-inspector-fallback-stub", "docling-stub"],
        )
        parse_component = injector.get(ParseComponent)
        file_info = get_file_info(
            TEST_FOLDER_PATH.parents[0] / "readers" / "files" / "file_scanned.pdf",
            "file_scanned.pdf",
        )
        result = parse_component.file_to_nodes(file_info)
    finally:
        reader_component.register_extension_readers(".pdf", original_reader_names)
        reader_component.unregister_reader_factory("pdf-inspector-fallback-stub")
        reader_component.unregister_reader_factory("docling-stub")

    assert result.reader == "docling-stub"
    assert len(result.nodes) == 1
    assert result.nodes[0].get_content(MetadataMode.NONE) == "docling content"


def test_pdf_inspector_fallback_without_next_reader_raises(
    injector: MockInjector,
) -> None:
    from private_gpt.artifact_index.artifact_exception import InvalidFileError

    injector.bind_settings(
        {
            "data": {
                "reader": "auto",
                "local_data_folder": "local_data/tests",
                "enable_vision_fallback": False,
            }
        }
    )
    reader_component = injector.get(ReaderComponent)
    original_reader_names = reader_component.get_reader_names(extension=".pdf")

    reader_component.register_reader_factory(
        "pdf-inspector-fallback-stub",
        _FallbackReaderFactory(injector.get(Settings), injector.test_injector),
    )
    try:
        reader_component.register_extension_readers(
            ".pdf",
            ["pdf-inspector-fallback-stub"],
        )
        parse_component = injector.get(ParseComponent)
        file_info = get_file_info(
            TEST_FOLDER_PATH.parents[0] / "readers" / "files" / "file_scanned.pdf",
            "file_scanned.pdf",
        )
        with pytest.raises(InvalidFileError):
            parse_component.file_to_nodes(file_info)
    finally:
        reader_component.register_extension_readers(".pdf", original_reader_names)
        reader_component.unregister_reader_factory("pdf-inspector-fallback-stub")
