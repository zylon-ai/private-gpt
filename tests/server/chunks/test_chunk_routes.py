from pathlib import Path

from fastapi.testclient import TestClient

from private_gpt.server.chunks.chunks_router import ChunksBody, ChunksResponse
from tests.fixtures.ingest_helper import IngestHelper


def test_chunks_retrieval(test_client: TestClient, ingest_helper: IngestHelper) -> None:
    # Make sure there is at least some chunk to query in the database
    path = Path(__file__).parents[0] / "chunk_test.txt"
    ingest_helper.ingest_file(path)

    body = ChunksBody(text="b483dd15-78c4-4d67-b546-21a0d690bf43")
    response = test_client.post("/v1/chunks", json=body.model_dump())
    assert response.status_code == 200
    chunk_response = ChunksResponse.model_validate(response.json())
    assert len(chunk_response.data) > 0
