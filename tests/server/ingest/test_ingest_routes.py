from pathlib import Path

from tests.fixtures.ingest_helper import IngestHelper


def test_ingest_accepts_txt_files(ingest_helper: IngestHelper) -> None:
    path = Path(__file__).parents[0] / "test.txt"
    ingest_result = ingest_helper.ingest_file(path)
    assert len(ingest_result.data) == 1


def test_ingest_accepts_pdf_files(ingest_helper: IngestHelper) -> None:
    path = Path(__file__).parents[0] / "test.pdf"
    ingest_result = ingest_helper.ingest_file(path)
    assert len(ingest_result.data) == 1
