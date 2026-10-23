import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from llama_index.core.schema import BaseNode, Document, MetadataMode

from private_gpt.components.ingest.metadata_helper import MetadataKeys
from private_gpt.components.ingest.utils import get_file_info
from private_gpt.components.readers.base_reader import IngestionReader
from private_gpt.components.readers.factories.base import ReaderFactory
from private_gpt.components.readers.reader_component import ReaderComponent
from private_gpt.settings.settings import Settings
from tests.fixtures.mock_injector import MockInjector

TEST_FOLDER_PATH = Path(__file__).parents[0]
TEST_FILE_PATH = TEST_FOLDER_PATH / "files"
mock_extra_info = {
    MetadataKeys.ARTIFACT_ID.value: str(uuid.uuid4()),
    MetadataKeys.COLLECTION.value: str(uuid.uuid4()),
}


async def collect_nodes(
    file_info: Any,
    reader: IngestionReader,
) -> list[BaseNode]:
    nodes: list[BaseNode] = []
    async for node in reader.lazy_load_data(
        file_info=file_info,
        extra_info=mock_extra_info,
    ):
        nodes.append(node)
    return nodes


class StubReader(IngestionReader):
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
            text="stub reader content",
            extra_info=extra_info if extra_info is not None else {},
        )


class StubReaderFactory(ReaderFactory):
    def create_reader(self, extension: str | None = None) -> IngestionReader:
        del extension
        return StubReader()


class FailingReader(IngestionReader):
    async def lazy_load_data(
        self,
        file_info: Any,
        extra_info: dict[str, Any] | None = None,
        execute_transformations: bool = True,
        *args: Any,
        **load_kwargs: Any,
    ) -> AsyncIterator[BaseNode]:
        del file_info, extra_info, execute_transformations, args, load_kwargs
        if False:
            yield Document(text="")
        raise RuntimeError("reader failure")


class FailingReaderFactory(ReaderFactory):
    def create_reader(self, extension: str | None = None) -> IngestionReader:
        del extension
        return FailingReader()


class ImportFailingReaderFactory(ReaderFactory):
    def create_reader(self, extension: str | None = None) -> IngestionReader:
        del extension
        raise ImportError("reader import failure")


def test_extension_registry_can_be_swapped_at_runtime(injector: MockInjector) -> None:
    injector.bind_settings(
        {
            "data": {
                "reader": "auto",
                "local_data_folder": "local_data/tests",
            }
        }
    )
    reader_component = injector.get(ReaderComponent)
    original_reader_name = reader_component.get_registered_reader_name(".txt")
    assert original_reader_name is not None

    reader_component.register_reader_factory(
        "stub-reader",
        StubReaderFactory(injector.get(Settings), injector.test_injector),
    )
    try:
        reader_component.register_extension_reader(".txt", "stub-reader")
        reader = reader_component.get_reader_by_extension(".txt")
        nodes = asyncio.run(
            collect_nodes(
                get_file_info(TEST_FILE_PATH / "file.txt", "file.txt"),
                reader,
            )
        )
    finally:
        reader_component.register_extension_reader(".txt", original_reader_name)
        reader_component.unregister_reader_factory("stub-reader")

    assert len(nodes) == 1
    assert nodes[0].get_content(MetadataMode.NONE) == "stub reader content"


@pytest.mark.asyncio
async def test_auto_reader_falls_back_to_next_candidate_on_import_error(
    injector: MockInjector,
) -> None:
    injector.bind_settings(
        {
            "data": {
                "reader": "auto",
                "local_data_folder": "local_data/tests",
            }
        }
    )
    reader_component = injector.get(ReaderComponent)
    original_reader_names = reader_component.get_reader_names(extension=".txt")

    reader_component.register_reader_factory(
        "import-failing-reader",
        ImportFailingReaderFactory(injector.get(Settings), injector.test_injector),
    )
    reader_component.register_reader_factory(
        "stub-reader",
        StubReaderFactory(injector.get(Settings), injector.test_injector),
    )
    try:
        reader_component.register_extension_readers(
            ".txt",
            ["import-failing-reader", "stub-reader"],
        )
        nodes = await reader_component.load_data(
            get_file_info(TEST_FILE_PATH / "file.txt", "file.txt"),
            extra_info=mock_extra_info,
        )
    finally:
        reader_component.register_extension_readers(".txt", original_reader_names)
        reader_component.unregister_reader_factory("import-failing-reader")
        reader_component.unregister_reader_factory("stub-reader")

    assert len(nodes) == 1
    assert nodes[0].get_content(MetadataMode.NONE) == "stub reader content"


@pytest.mark.asyncio
async def test_auto_reader_does_not_fall_back_on_parse_error(
    injector: MockInjector,
) -> None:
    injector.bind_settings(
        {
            "data": {
                "reader": "auto",
                "local_data_folder": "local_data/tests",
            }
        }
    )
    reader_component = injector.get(ReaderComponent)
    original_reader_names = reader_component.get_reader_names(extension=".txt")

    reader_component.register_reader_factory(
        "failing-reader",
        FailingReaderFactory(injector.get(Settings), injector.test_injector),
    )
    reader_component.register_reader_factory(
        "stub-reader",
        StubReaderFactory(injector.get(Settings), injector.test_injector),
    )
    try:
        reader_component.register_extension_readers(
            ".txt",
            ["failing-reader", "stub-reader"],
        )
        with pytest.raises(RuntimeError, match="reader failure"):
            await reader_component.load_data(
                get_file_info(TEST_FILE_PATH / "file.txt", "file.txt"),
                extra_info=mock_extra_info,
            )
    finally:
        reader_component.register_extension_readers(".txt", original_reader_names)
        reader_component.unregister_reader_factory("failing-reader")
        reader_component.unregister_reader_factory("stub-reader")


def test_markitdown_reader_supports_docx(injector: MockInjector) -> None:
    pytest.importorskip("markitdown")

    file = TEST_FILE_PATH / "file.docx"
    mock_file_info = get_file_info(file, "file.docx")

    reader = injector.get(ReaderComponent).get_reader("markitdown", extension=".docx")
    nodes = asyncio.run(collect_nodes(mock_file_info, reader))

    assert len(nodes) > 1

    markdown_content = "".join([node.get_content(MetadataMode.NONE) for node in nodes])
    assert "Mission, Vision, Values" in markdown_content


def test_markitdown_reader_can_be_selected_by_name(injector: MockInjector) -> None:
    pytest.importorskip("markitdown")

    file = TEST_FILE_PATH / "file.xlsx"
    mock_file_info = get_file_info(file, "file.xlsx")

    reader = injector.get(ReaderComponent).get_reader("markitdown", extension=".xlsx")
    nodes = asyncio.run(collect_nodes(mock_file_info, reader))

    markdown_content = "".join([node.get_content(MetadataMode.NONE) for node in nodes])
    assert "## Plan1" in markdown_content


def test_forced_reader_setting_overrides_auto(injector: MockInjector) -> None:
    injector.bind_settings(
        {
            "data": {
                "reader": "markitdown",
                "local_data_folder": "local_data/tests",
            }
        }
    )

    reader_component = injector.get(ReaderComponent)
    assert reader_component.get_reader_names(extension=".docx") == ["markitdown"]
