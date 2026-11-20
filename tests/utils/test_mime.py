import sys
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from private_gpt.components.ingest import utils as ingest_utils
from private_gpt.utils import mime

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
