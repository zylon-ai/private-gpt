from fastapi.testclient import TestClient

from private_gpt.server.embeddings.embeddings_router import (
    EmbeddingsBody,
    EmbeddingsResponse,
)


def test_embeddings_generation(test_client: TestClient) -> None:
    body = EmbeddingsBody(input="Embed me")
    response = test_client.post("/v1/embeddings", json=body.model_dump())

    assert response.status_code == 200
    embedding_response = EmbeddingsResponse.model_validate(response.json())
    assert len(embedding_response.data) > 0
    assert len(embedding_response.data[0].embedding) > 0
