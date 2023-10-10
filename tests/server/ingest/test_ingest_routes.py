from pathlib import Path

from fastapi.testclient import TestClient

from private_gpt.server.ingest.routes import IngestResponse


def test_ingest_accepts_files(test_client: TestClient) -> None:
    path = Path(__file__).parents[0] / "test.txt"
    files = {"file": ("test.txt", path.open("rb"))}

    # Upload the file to fastapi using test_client
    response = test_client.post("/v1/ingest", files=files)

    # Not a very good test because we don't have a way to wipe ingestion beforehand
    # but at least it validates the service is working`
    assert response.status_code == 200

    ingest_result = IngestResponse.model_validate(response.json())
    assert len(ingest_result.documents) == 1
