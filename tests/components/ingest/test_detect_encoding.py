from pathlib import Path

import pytest
from chardet.universaldetector import UniversalDetector

from private_gpt.components.ingest.utils import detect_encoding


@pytest.fixture
def fed_bytes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record how many bytes each detector.feed() call receives."""
    sizes: list[int] = []
    original_feed = UniversalDetector.feed

    def counting_feed(self: UniversalDetector, byte_str: bytes) -> None:
        sizes.append(len(byte_str))
        original_feed(self, byte_str)

    monkeypatch.setattr(UniversalDetector, "feed", counting_feed)
    return sizes


def test_binary_file_is_not_scanned_to_the_end(
    tmp_path: Path, fed_bytes: list[int]
) -> None:
    # Shaped like a PDF: a text header, then binary streams with NUL bytes.
    file = tmp_path / "manual.pdf"
    file.write_bytes(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n" + bytes(range(256)) * 4096)

    assert detect_encoding(file) is None
    assert sum(fed_bytes) <= 4096


def test_utf16_text_is_still_detected(tmp_path: Path) -> None:
    file = tmp_path / "notes.txt"
    file.write_bytes(("Quarterly report: café, 第三季\n" * 500).encode("utf-16"))

    assert detect_encoding(file) == "UTF-16"


def test_text_without_nul_bytes_is_still_detected(tmp_path: Path) -> None:
    file = tmp_path / "notes.txt"
    file.write_bytes(
        ("これは四半期の売上に関する報告書です。\n" * 200).encode("shift_jis")
    )

    assert detect_encoding(file) == "SHIFT_JIS"
