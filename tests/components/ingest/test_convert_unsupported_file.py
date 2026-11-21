import shutil
from pathlib import Path

import pytest

from private_gpt.components.ingest import utils as ingest_utils

_LEGACY_WORD_BYTES = b"\xd0\xcf\x11\xe0 fake legacy word payload"


def _fake_libreoffice(
    file_info: ingest_utils.FileInfo,
    target_extension: str,
    raise_in_exception: bool = False,
) -> Path:
    """Stand-in for a successful ``soffice --convert-to`` run."""
    output = file_info.file_data.with_suffix(target_extension)
    shutil.copyfile(file_info.file_data, output)
    return output


def _convert(
    tmp_path: Path, source_name: str, monkeypatch: pytest.MonkeyPatch
) -> ingest_utils.FileInfo:
    monkeypatch.setattr(
        ingest_utils, "convert_file_with_libreoffice", _fake_libreoffice
    )
    source = tmp_path / source_name
    source.write_bytes(_LEGACY_WORD_BYTES)
    return ingest_utils.convert_unsupported_file(
        ingest_utils.get_file_info(source, source_name)
    )


def test_converted_file_takes_the_extension_of_the_converted_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converted = _convert(tmp_path, "QUARTERLY-REPORT.DOC", monkeypatch)

    assert converted.file_data.name == "QUARTERLY-REPORT.docx"
    assert converted.file_name == "QUARTERLY-REPORT.docx"
    assert converted.extension == ".docx"


def test_converted_file_name_keeps_the_extension_spelled_in_the_stem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    converted = _convert(tmp_path, "notes.doc.backup.doc", monkeypatch)

    assert converted.file_name == "notes.doc.backup.docx"
    assert converted.extension == ".docx"
