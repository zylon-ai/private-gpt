import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from private_gpt.server.ingest.ingest_router import IngestResponse
from tests.fixtures.ingest_helper import IngestHelper


def test_ingest_accepts_txt_files(ingest_helper: IngestHelper) -> None:
    path = Path(__file__).parents[0] / "test.txt"
    ingest_result = ingest_helper.ingest_file(path)
    assert len(ingest_result.data) == 1


def test_ingest_accepts_pdf_files(ingest_helper: IngestHelper) -> None:
    path = Path(__file__).parents[0] / "test.pdf"
    ingest_result = ingest_helper.ingest_file(path)
    assert len(ingest_result.data) == 1


def test_ingest_list_returns_something_after_ingestion(
    test_client: TestClient, ingest_helper: IngestHelper
) -> None:
    response_before = test_client.get("/v1/ingest/list")
    count_ingest_before = len(response_before.json()["data"])
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as test_file:
        test_file.write("Foo bar; hello there!")
        test_file.flush()
        test_file.seek(0)
        ingest_result = ingest_helper.ingest_file(Path(test_file.name))
    assert len(ingest_result.data) == 1, "The temp doc should have been ingested"
    response_after = test_client.get("/v1/ingest/list")
    count_ingest_after = len(response_after.json()["data"])
    assert (
        count_ingest_after == count_ingest_before + 1
    ), "The temp doc should be returned"


def test_ingest_plain_text(test_client: TestClient) -> None:
    response = test_client.post(
        "/v1/ingest/text", json={"file_name": "file_name", "text": "text"}
    )
    assert response.status_code == 200
    ingest_result = IngestResponse.model_validate(response.json())
    assert len(ingest_result.data) == 1
