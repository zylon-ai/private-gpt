import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from private_gpt.chat.extensions.context_filter import ContextFilter
from private_gpt.server.content.content_router import (
    ChunkedContentResponse,
    ContentBody,
    ContentResponse,
)
from tests.fixtures.ingest_helper import IngestHelper


def test_retrieve_content(test_client: TestClient, ingest_helper: IngestHelper) -> None:
    # Make sure there is at least some chunk to query in the database
    collection = str(uuid.uuid4())
    path = Path(__file__).parents[0] / "chunk_test.txt"
    ingest_response = ingest_helper.ingest_file(
        path, artifact=str(uuid.uuid4()), collection=collection
    )
    artifact_1 = ingest_response.data[0].artifact
    body = ContentBody(
        context_filter=ContextFilter(collection=collection),
    )
    response = test_client.post("/v1/artifacts/content", json=body.model_dump())
    assert response.status_code == 200
    chunk_response = ContentResponse.model_validate(response.json())
    assert len(chunk_response.data) == 1

    # Upload another file
    path = Path(__file__).parents[0] / "chunk_test.txt"
    ingest_response = ingest_helper.ingest_file(
        path, artifact=str(uuid.uuid4()), collection=collection
    )
    artifact_2 = ingest_response.data[0].artifact
    body = ContentBody(
        context_filter=ContextFilter(
            artifacts=[artifact_1, artifact_2], collection=collection
        ),
    )
    response = test_client.post("/v1/artifacts/content", json=body.model_dump())
    assert response.status_code == 200
    chunk_response = ContentResponse.model_validate(response.json())
    assert len(chunk_response.data) == 2


def test_retrieve_chunked_content(
    test_client: TestClient, ingest_helper: IngestHelper
) -> None:
    # Make sure there is at least some chunk to query in the database
    collection = str(uuid.uuid4())
    path = Path(__file__).parents[0] / "chunk_test.txt"
    ingest_response = ingest_helper.ingest_file(
        path, artifact=str(uuid.uuid4()), collection=collection
    )
    artifact_1 = ingest_response.data[0].artifact
    body = ContentBody(
        context_filter=ContextFilter(collection=collection),
    )
    response = test_client.post("/v1/artifacts/chunked-content", json=body.model_dump())
    assert response.status_code == 200
    chunk_response = ChunkedContentResponse.model_validate(response.json())
    assert len(chunk_response.data) == 1
    first_element = chunk_response.data[0]
    assert len(first_element.content) == 2

    # Upload another file
    path = Path(__file__).parents[0] / "chunk_test.txt"
    ingest_response = ingest_helper.ingest_file(
        path, artifact=str(uuid.uuid4()), collection=collection
    )
    artifact_2 = ingest_response.data[0].artifact
    body = ContentBody(
        context_filter=ContextFilter(
            artifacts=[artifact_1, artifact_2], collection=collection
        ),
        max_tokens=10,
    )
    response = test_client.post("/v1/artifacts/chunked-content", json=body.model_dump())
    assert response.status_code == 200
    chunk_response = ChunkedContentResponse.model_validate(response.json())
    assert len(chunk_response.data) == 2
    first_element = chunk_response.data[0]
    assert len(first_element.content) == 2
