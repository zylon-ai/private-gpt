import asyncio
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from llama_index.core.schema import MetadataMode

from private_gpt.components.ingest.metadata_helper import MetadataKeys
from private_gpt.components.ingest.utils import get_file_info
from private_gpt.components.readers.pdf_inspector.pdf_inspector_reader import (
    PdfInspectorFallbackError,
    PdfInspectorReader,
)

pytest.importorskip("pdf_inspector")

TEST_FOLDER_PATH = Path(__file__).parents[0]
TEST_FILE_PATH = TEST_FOLDER_PATH / "files"
mock_extra_info = {
    MetadataKeys.ARTIFACT_ID.value: str(uuid.uuid4()),
    MetadataKeys.COLLECTION.value: str(uuid.uuid4()),
}


class _FakeResult:
    def __init__(
        self,
        pdf_type: str,
        confidence: float = 1.0,
        pages_needing_ocr: list[int] | None = None,
        markdown: str | None = "some markdown",
    ) -> None:
        self.pdf_type = pdf_type
        self.confidence = confidence
        self.pages_needing_ocr = pages_needing_ocr or []
        self.markdown = markdown


def test_pdf_inspector_extracts_text_based_pdf() -> None:
    file = TEST_FILE_PATH / "file_general_elements.pdf"
    mock_file_info = get_file_info(file, "file_general_elements.pdf")

    reader = PdfInspectorReader()
    nodes = asyncio.run(_collect_nodes(mock_file_info, reader))

    assert len(nodes) >= 1
    content = "".join(node.get_content(MetadataMode.NONE) for node in nodes)
    assert content.strip()


def test_pdf_inspector_falls_back_on_scanned_pdf() -> None:
    file = TEST_FILE_PATH / "file_scanned.pdf"
    mock_file_info = get_file_info(file, "file_scanned.pdf")

    reader = PdfInspectorReader()
    with pytest.raises(PdfInspectorFallbackError):
        asyncio.run(_collect_nodes(mock_file_info, reader))


def test_pdf_inspector_falls_back_on_pages_needing_ocr() -> None:
    fake_result = _FakeResult(pdf_type="text_based", pages_needing_ocr=[2])
    with patch(
        "pdf_inspector.process_pdf",
        return_value=fake_result,
    ):
        reader = PdfInspectorReader()
        mock_file_info = get_file_info(
            TEST_FILE_PATH / "file_general_elements.pdf",
            "file_general_elements.pdf",
        )
        with pytest.raises(PdfInspectorFallbackError):
            asyncio.run(_collect_nodes(mock_file_info, reader))


def test_pdf_inspector_falls_back_on_low_confidence_mixed() -> None:
    fake_result = _FakeResult(pdf_type="mixed", confidence=0.3)
    with patch(
        "pdf_inspector.process_pdf",
        return_value=fake_result,
    ):
        reader = PdfInspectorReader()
        mock_file_info = get_file_info(
            TEST_FILE_PATH / "file_general_elements.pdf",
            "file_general_elements.pdf",
        )
        with pytest.raises(PdfInspectorFallbackError):
            asyncio.run(_collect_nodes(mock_file_info, reader))


def test_pdf_inspector_extracts_high_confidence_mixed() -> None:
    fake_result = _FakeResult(
        pdf_type="mixed", confidence=0.9, markdown="mixed content"
    )
    with patch(
        "pdf_inspector.process_pdf",
        return_value=fake_result,
    ):
        reader = PdfInspectorReader()
        mock_file_info = get_file_info(
            TEST_FILE_PATH / "file_general_elements.pdf",
            "file_general_elements.pdf",
        )
        nodes = asyncio.run(_collect_nodes(mock_file_info, reader))

    assert len(nodes) >= 1


def test_pdf_inspector_falls_back_on_exception() -> None:
    with patch(
        "pdf_inspector.process_pdf",
        side_effect=RuntimeError("boom"),
    ):
        reader = PdfInspectorReader()
        mock_file_info = get_file_info(
            TEST_FILE_PATH / "file_general_elements.pdf",
            "file_general_elements.pdf",
        )
        with pytest.raises(PdfInspectorFallbackError):
            asyncio.run(_collect_nodes(mock_file_info, reader))


async def _collect_nodes(file_info: Any, reader: PdfInspectorReader) -> list[Any]:
    nodes = []
    async for node in reader.lazy_load_data(
        file_info=file_info,
        extra_info=mock_extra_info,
    ):
        nodes.append(node)
    return nodes
