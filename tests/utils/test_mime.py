import io
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from private_gpt.components.ingest import utils as ingest_utils
from private_gpt.components.ingest.ingest_helper import IngestionHelper
from private_gpt.components.ingest.utils import FileInfo
from private_gpt.utils import mime

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
DOCM = "application/vnd.ms-word.document.macroEnabled.12"
ODT = "application/vnd.oasis.opendocument.text"

PDF = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<>>\nendobj\n"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    # On Windows libmagic hangs inside magic.from_buffer/from_file; any call
    # into it would block the request forever.
    hanging = MagicMock(side_effect=AssertionError("libmagic called on Windows"))
    fake = MagicMock(from_buffer=hanging, from_file=hanging)
    fake.Magic.return_value = fake
    monkeypatch.setitem(sys.modules, "magic", fake)
    monkeypatch.setattr(mime.sys, "platform", "win32")
    mime.is_magic_available.cache_clear()
    yield fake
    mime.is_magic_available.cache_clear()


def test_magic_unavailable_on_windows(windows: MagicMock) -> None:
    assert mime.is_magic_available() is False


def test_buffer_detection_on_windows_sniffs_content(windows: MagicMock) -> None:
    assert mime.detect_mime_from_buffer(PDF) == "application/pdf"
    assert mime.detect_mime_from_buffer(PNG) == "image/png"
    assert mime.detect_mime_from_buffer(b"plain words") == "application/octet-stream"


def test_file_detection_on_windows_uses_content_then_extension(
    windows: MagicMock, tmp_path: Path
) -> None:
    pdf = tmp_path / "report.bin"
    pdf.write_bytes(PDF)
    text = tmp_path / "notes.txt"
    text.write_text("hello")

    assert mime.detect_mime_from_file(pdf) == "application/pdf"
    assert mime.detect_mime_from_file(text) == "text/plain"


def test_ingest_actual_mime_type_on_windows_skips_libmagic(
    windows: MagicMock, tmp_path: Path
) -> None:
    # Same as a host without libmagic: no content type, so validation falls
    # back to the extension instead of reporting a mismatch.
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(PDF)

    assert ingest_utils.get_actual_mime_type(pdf) is None


def _package(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)
    return buffer.getvalue()


def _content_types(part: str, content_type: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        f'<Override PartName="{part}" ContentType="{content_type}"/>'
        "</Types>"
    ).encode()


def _docx_with_leading_trash() -> bytes:
    # Word writes discarded parts before word/ and [Content_Types].xml.
    # libmagic and filetype then stop at application/zip.
    return _package(
        {
            "[trash]/0000.dat": b"\x00" * 4000,
            "[trash]/0001.dat": b"\x00" * 2000,
            "word/document.xml": b"<w:document/>",
            "[Content_Types].xml": _content_types(
                "/word/document.xml",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml",
            ),
        }
    )


def test_office_packages_are_not_reported_as_zip(windows: MagicMock) -> None:
    docx = _docx_with_leading_trash()
    xlsx = _package(
        {
            "xl/workbook.xml": b"<workbook/>",
            "[Content_Types].xml": _content_types(
                "/xl/workbook.xml",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml",
            ),
        }
    )
    pptx = _package(
        {
            "ppt/presentation.xml": b"<presentation/>",
            "[Content_Types].xml": _content_types(
                "/ppt/presentation.xml",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml",
            ),
        }
    )
    docm = _package(
        {
            "word/document.xml": b"<w:document/>",
            "word/vbaProject.bin": b"vba",
            "[Content_Types].xml": _content_types(
                "/word/document.xml",
                "application/vnd.ms-word.document.macroEnabled.main+xml",
            ),
        }
    )
    odt = _package(
        {
            "content.xml": b"<office:document-content/>",
            "mimetype": b"application/vnd.oasis.opendocument.text",
        }
    )

    assert mime.detect_mime_from_buffer(docx) == DOCX
    assert mime.detect_mime_from_buffer(xlsx) == XLSX
    assert mime.detect_mime_from_buffer(pptx) == PPTX
    assert mime.detect_mime_from_buffer(docm) == DOCM
    assert mime.detect_mime_from_buffer(odt) == ODT
    assert windows.Magic.call_count == 0


def test_plain_zip_stays_zip(windows: MagicMock) -> None:
    archive = _package({"readme.txt": b"hello", "docs/word/document.xml": b"<w/>"})
    assert mime.detect_mime_from_buffer(archive) == "application/zip"
    assert (
        mime.detect_mime_from_buffer(b"PK\x03\x04" + b"\x00" * 64) == "application/zip"
    )


def test_office_package_detected_from_path_without_extension(
    windows: MagicMock, tmp_path: Path
) -> None:
    path = tmp_path / "upload.bin"
    path.write_bytes(_docx_with_leading_trash())

    assert mime.detect_mime_from_file(path) == DOCX


def test_actual_mime_type_refines_zip_sniff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "memo.docx"
    path.write_bytes(_docx_with_leading_trash())
    fake = MagicMock()
    fake.Magic.return_value.from_file.return_value = "application/zip"
    monkeypatch.setitem(sys.modules, "magic", fake)
    monkeypatch.setattr(mime, "is_magic_available", lambda: True)

    assert ingest_utils.get_actual_mime_type(path) == DOCX

    file_info = FileInfo(
        file_name="memo.docx",
        extension=".docx",
        file_data=path,
        guest_mime_type=DOCX,
        actual_mime_type=ingest_utils.get_actual_mime_type(path),
        file_size=path.stat().st_size,
    )
    errors, warnings = IngestionHelper.validate_file_info(file_info)
    assert errors == []
    assert warnings == []
