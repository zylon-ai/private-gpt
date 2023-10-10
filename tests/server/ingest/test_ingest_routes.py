from pathlib import Path

from fastapi.testclient import TestClient

from private_gpt.server.ingest.routes import IngestResponse


def test_ingest_accepts_txt_files(test_client: TestClient) -> None:
    path = Path(__file__).parents[0] / "test.txt"
    files = {"file": ("test.txt", path.open("rb"))}
    response = test_client.post("/v1/ingest", files=files)

    assert response.status_code == 200
    ingest_result = IngestResponse.model_validate(response.json())
    assert len(ingest_result.documents) == 1


def test_ingest_accepts_pdf_files(test_client: TestClient) -> None:
    path = Path(__file__).parents[0] / "test.pdf"
    files = {"file": ("test.pdf", path.open("rb"))}
    response = test_client.post("/v1/ingest", files=files)

    assert response.status_code == 200
    ingest_result = IngestResponse.model_validate(response.json())
    assert len(ingest_result.documents) == 1
