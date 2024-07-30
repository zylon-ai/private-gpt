from fastapi.testclient import TestClient

from private_gpt.server.recipes.summarize.summarize_router import (
    SummarizeBody,
    SummarizeResponse,
)


def test_summarize_route_produces_a_stream(test_client: TestClient) -> None:
    body = SummarizeBody(
        text="Test",
        stream=True,
    )
    response = test_client.post("/v1/summarize", json=body.model_dump())

    raw_events = response.text.split("\n\n")
    events = [
        item.removeprefix("data: ") for item in raw_events if item.startswith("data: ")
    ]
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert len(events) > 0
    assert events[-1] == "[DONE]"


def test_summarize_route_produces_a_single_value(test_client: TestClient) -> None:
    body = SummarizeBody(
        text="test",
        stream=False,
    )
    response = test_client.post("/v1/summarize", json=body.model_dump())

    # No asserts, if it validates it's good
    SummarizeResponse.model_validate(response.json())
    assert response.status_code == 200


def test_summarize_with_document_context(test_client: TestClient) -> None:
    # Ingest an document
    ingest_response = test_client.post(
        "/v1/ingest/text",
        json={
            "file_name": "file_name",
            "text": "Lorem ipsum dolor sit amet",
        },
    )
    assert ingest_response.status_code == 200
    ingested_docs = ingest_response.json()["data"]
    assert len(ingested_docs) == 1

    body = SummarizeBody(
        use_context=True,
        context_filter={"docs_ids": [doc["doc_id"] for doc in ingested_docs]},
        stream=False,
    )
    response = test_client.post("/v1/summarize", json=body.model_dump())

    completion: SummarizeResponse = SummarizeResponse.model_validate(response.json())
    assert response.status_code == 200
    # We can check the content of the completion, because mock LLM used in tests
    # always echoes the prompt. In the case of summary, the input context is passed.
    assert completion.summary.find("Lorem ipsum dolor sit amet") != -1


def test_summarize_with_non_existent_document_context_not_fails(
    test_client: TestClient,
) -> None:
    body = SummarizeBody(
        use_context=True,
        context_filter={
            "docs_ids": ["non-existent-doc-id"],
        },
        stream=False,
    )

    response = test_client.post("/v1/summarize", json=body.model_dump())

    completion: SummarizeResponse = SummarizeResponse.model_validate(response.json())
    assert response.status_code == 200
    # We can check the content of the completion, because mock LLM used in tests
    # always echoes the prompt. In the case of summary, the input context is passed.
    assert completion.summary.find("Empty Response") != -1


def test_summarize_with_metadata_and_document_context(test_client: TestClient) -> None:
    docs = []

    # Ingest a first document
    document_1_content = "Content of document 1"
    ingest_response = test_client.post(
        "/v1/ingest/text",
        json={
            "file_name": "file_name_1",
            "text": document_1_content,
        },
    )
    assert ingest_response.status_code == 200
    ingested_docs = ingest_response.json()["data"]
    assert len(ingested_docs) == 1
    docs += ingested_docs

    # Ingest a second document
    document_2_content = "Text of document 2"
    ingest_response = test_client.post(
        "/v1/ingest/text",
        json={
            "file_name": "file_name_2",
            "text": document_2_content,
        },
    )
    assert ingest_response.status_code == 200
    ingested_docs = ingest_response.json()["data"]
    assert len(ingested_docs) == 1
    docs += ingested_docs

    # Completions with the first document's id and the second document's metadata
    body = SummarizeBody(
        use_context=True,
        context_filter={"docs_ids": [doc["doc_id"] for doc in docs]},
        stream=False,
    )
    response = test_client.post("/v1/summarize", json=body.model_dump())

    completion: SummarizeResponse = SummarizeResponse.model_validate(response.json())
    assert response.status_code == 200
    # Assert both documents are part of the used sources
    # We can check the content of the completion, because mock LLM used in tests
    # always echoes the prompt. In the case of summary, the input context is passed.
    assert completion.summary.find(document_1_content) != -1
    assert completion.summary.find(document_2_content) != -1


def test_summarize_with_prompt(test_client: TestClient) -> None:
    ingest_response = test_client.post(
        "/v1/ingest/text",
        json={
            "file_name": "file_name",
            "text": "Lorem ipsum dolor sit amet",
        },
    )
    assert ingest_response.status_code == 200
    ingested_docs = ingest_response.json()["data"]
    assert len(ingested_docs) == 1

    body = SummarizeBody(
        use_context=True,
        context_filter={
            "docs_ids": [doc["doc_id"] for doc in ingested_docs],
        },
        prompt="This is a custom summary prompt, 54321",
        stream=False,
    )
    response = test_client.post("/v1/summarize", json=body.model_dump())

    completion: SummarizeResponse = SummarizeResponse.model_validate(response.json())
    assert response.status_code == 200
    # We can check the content of the completion, because mock LLM used in tests
    # always echoes the prompt. In the case of summary, the input context is passed.
    assert completion.summary.find("This is a custom summary prompt, 54321") != -1
