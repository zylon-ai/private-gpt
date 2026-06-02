from pathlib import Path

from fastapi.testclient import TestClient

from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.server.primitives.primitives_service import (
    SearchResponse,
    SemanticSearch,
)
from tests.fixtures.ingest_helper import IngestHelper


def test_chunks_retrieval(test_client: TestClient, ingest_helper: IngestHelper) -> None:
    # Make sure there is at least some chunk to query in the database
    path = Path(__file__).parents[0] / "chunk_test.txt"
    ingest_helper.ingest_file(path, collection="chunk_collection")

    body = SemanticSearch(
        text="b483dd15-78c4-4d67-b546-21a0d690bf43",
        context_filter=ContextFilter(collection="chunk_collection"),
        expand=False,
    )
    response = test_client.post("/v1/primitives/search", json=body.model_dump())
    assert response.status_code == 200
    chunk_response = SearchResponse.model_validate(response.json())
    assert len(chunk_response.data) > 0
    for chunk in chunk_response.data:
        assert chunk.object == "context.chunk"
        assert len(chunk.previous_texts) == 0
        assert len(chunk.next_texts) == 0


def test_chunks_retrieval_with_expansion(
    test_client: TestClient, ingest_helper: IngestHelper
) -> None:
    # Make sure there is at least some chunk to query in the database
    path = Path(__file__).parents[0] / "chunk_test.txt"
    ingest_helper.ingest_file(path, collection="chunk_collection")

    body = SemanticSearch(
        text="b483dd15-78c4-4d67-b546-21a0d690bf43",
        context_filter=ContextFilter(collection="chunk_collection"),
        expand=True,
    )
    response = test_client.post("/v1/primitives/search", json=body.model_dump())
    assert response.status_code == 200
    chunk_response = SearchResponse.model_validate(response.json())
    assert len(chunk_response.data) > 0


def test_chunks_retrieval_without_query(
    test_client: TestClient, ingest_helper: IngestHelper
) -> None:
    # Make sure there is at least some chunk to query in the database
    path = Path(__file__).parents[0] / "chunk_test.txt"
    ingest_helper.ingest_file(path, collection="chunk_collection")

    body = SemanticSearch(
        text="",
        context_filter=ContextFilter(collection="chunk_collection"),
        expand=True,
    )
    response = test_client.post("/v1/primitives/search", json=body.model_dump())
    assert response.status_code == 200
    chunk_response = SearchResponse.model_validate(response.json())
    assert len(chunk_response.data) == 0
