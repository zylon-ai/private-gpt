from fastapi.testclient import TestClient


def test_default_does_not_require_auth(test_client: TestClient) -> None:
    response_before = test_client.get("/v1/ingest/list")
    assert response_before.status_code == 200
