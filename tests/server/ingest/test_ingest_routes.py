import tempfile
import uuid
from pathlib import Path
from random import random
from unittest.mock import Mock

from fastapi.testclient import TestClient

from private_gpt.components.broker.broker_component import BrokerComponent
from private_gpt.components.ingestion.ingestion_scheduler import (
    IngestionSchedulerFactory,
)
from private_gpt.components.storage.s3_helper import S3Helper
from private_gpt.di import set_global_injector
from private_gpt.server.ingest.ingest_router import (
    DeleteIngestedDocumentAsyncBody,
    DeleteIngestedDocumentBody,
    IngestAsyncBody,
    IngestBody,
    IngestResponse,
)
from private_gpt.server.utils.artifact_input import UriArtifact
from private_gpt.server.utils.callback import AMQP, AsyncResponse, Callback
from tests.fixtures.ingest_helper import IngestHelper
from tests.fixtures.mock_injector import MockInjector


def _use_celery_ingestion(injector: MockInjector) -> None:
    settings = injector.bind_settings({"scheduler": {"ingestion": {"mode": "celery"}}})
    factory = IngestionSchedulerFactory(
        settings=settings,
        injector=injector.test_injector,
    )
    injector.bind_mock(IngestionSchedulerFactory, factory)


def test_ingest_accepts_txt_files(
    test_client: TestClient, ingest_helper: IngestHelper
) -> None:
    collection = str(uuid.uuid4())
    path = Path(__file__).parents[0] / "test.txt"
    ingest_result = ingest_helper.ingest_file(path, collection=collection)
    assert len(ingest_result.data) == 1

    # Delete the created temp file
    ingest_helper.delete_file(collection, "test.txt")


def test_ingest_list_returns_something_after_ingestion(
    test_client: TestClient, ingest_helper: IngestHelper
) -> None:
    collection = str(uuid.uuid4())
    response_before = test_client.get(f"/v1/artifacts/list?collection={collection}")
    count_ingest_before = len(response_before.json()["data"])
    with tempfile.NamedTemporaryFile("w", suffix=".txt") as test_file:
        test_file.write("Foo bar; hello there!")
        test_file.flush()
        test_file.seek(0)
        ingest_result = ingest_helper.ingest_file(
            Path(test_file.name), collection=collection
        )
        assert len(ingest_result.data) == 1, "The temp doc should have been ingested"
        response_after = test_client.get(f"/v1/artifacts/list?collection={collection}")
        count_ingest_after = len(response_after.json()["data"])
        assert count_ingest_after == count_ingest_before + 1, (
            "The temp doc should be returned"
        )

        # Delete the created temp file
        ingest_helper.delete_file(collection, Path(test_file.name).name)


def test_ingest_plain_text(
    test_client: TestClient, ingest_helper: IngestHelper
) -> None:
    collection = str(uuid.uuid4())
    response = test_client.post(
        "/v1/artifacts/ingest",
        json={
            "metadata": {},
            "input": {
                "type": "text",
                "value": "text",
            },
            "collection": collection,
            "artifact": "artifact_id",
        },
    )
    assert response.status_code == 200
    ingest_result = IngestResponse.model_validate(response.json())
    assert len(ingest_result.data) == 1

    # Delete the created temp file
    ingest_helper.delete_file(collection, "artifact_id")


def test_ingest_empty_text(test_client: TestClient) -> None:
    collection = str(uuid.uuid4())
    response = test_client.post(
        "/v1/artifacts/ingest",
        json={
            "metadata": {},
            "input": {
                "type": "text",
                "value": "",
            },
            "collection": collection,
            "artifact": "artifact_id",
        },
    )
    assert response.status_code == 400, "Empty text should not be accepted"


def test_ingest_uri_async(
    test_client: TestClient, injector: MockInjector, ingest_helper: IngestHelper
) -> None:
    collection = str(uuid.uuid4())
    _use_celery_ingestion(injector=injector)

    # Mock broker to receive callback
    broker_mock = Mock(BrokerComponent)
    injector.bind_mock(BrokerComponent, broker_mock)

    # Mock AWS S3 Helper
    s3_helper = Mock(S3Helper)
    injector.bind_mock(S3Helper, s3_helper)

    set_global_injector(injector.test_injector)

    path = Path(__file__).parents[0] / "test.txt"
    ingest_uri_body = IngestBody(
        input=UriArtifact(value=str(path)),
        metadata={"file_name": "test.txt"},
        collection=collection,
        artifact="artifact_id",
    )
    body = IngestAsyncBody(
        ingest_body=ingest_uri_body,
        callback=Callback(
            amqp=AMQP(
                exchange="main",
                routing_key_done="ingest.done",
                routing_key_progress="ingest.progress",
                routing_key_error="ingest.error",
            ),
            properties={"test": "123"},
        ),
    )

    response = test_client.post("/v1/artifacts/ingest/async", json=body.model_dump())

    assert response.status_code == 200
    content = response.json()
    task_id = content["task_id"]
    assert task_id

    response = test_client.get(f"/v1/artifacts/ingest/async/{task_id}")
    # Response will already contain the result as we are running tests synchronously
    assert response.status_code == 200
    content = response.json()
    assert content["task_id"] == task_id
    assert content["task_status"] == "SUCCESS"
    ingest_result = IngestResponse.model_validate(content["task_result"])
    assert len(ingest_result.data) == 1

    # Check if broker was called with the callback
    expected_response = AsyncResponse(
        data=ingest_result,
        error=None,
        type="pgpt.vector_index_task.done",
        callback_properties={"test": "123"},
    )

    broker_mock.publish.assert_called_with(
        exchange="main",
        routing_key="ingest.done",
        body=bytes(expected_response.model_dump_json(), "utf-8"),
    )

    # Delete the created temp file
    ingest_helper.delete_file(collection, "artifact_id")


def test_reingest_same_file_and_same_artifact(ingest_helper: IngestHelper) -> None:
    collection = str(uuid.uuid4())
    path = Path(__file__).parents[0] / "test.txt"
    ingest_result = ingest_helper.ingest_file(
        path, collection=collection, artifact="test.txt"
    )
    assert len(ingest_result.data) == 1

    ingest_result = ingest_helper.ingest_file(
        path, collection=collection, artifact="test.txt"
    )
    assert len(ingest_result.data) == 0

    # Delete the created temp file
    ingest_helper.delete_file(collection, "test.txt")


def test_reingest_same_file_and_different_artifact(ingest_helper: IngestHelper) -> None:
    collection = str(uuid.uuid4())
    path = Path(__file__).parents[0] / "test.txt"

    ingest_result = ingest_helper.ingest_file(
        path, collection=collection, artifact="test.txt"
    )
    assert len(ingest_result.data) == 1

    ingest_result = ingest_helper.ingest_file(
        path, collection=collection, artifact="new_artifact"
    )
    assert len(ingest_result.data) == 1

    # Delete the created temp files
    ingest_helper.delete_file(collection, "test.txt")
    ingest_helper.delete_file(collection, "new_artifact")


def test_list_metadata(test_client: TestClient, ingest_helper: IngestHelper) -> None:
    collection = str(uuid.uuid4())
    # Ingest a file
    random_metadata = random()
    test_client.post(
        "/v1/artifacts/ingest",
        json={
            "metadata": {"metadata_key": random_metadata},
            "input": {
                "type": "text",
                "value": "text",
            },
            "collection": collection,
            "artifact": "random_metadata",
        },
    )

    response = test_client.get(f"/v1/artifacts/list?collection={collection}")
    assert response.status_code == 200
    content = response.json()

    ingested_found = any(
        item.get("doc_metadata", {}).get("metadata_key") == random_metadata
        for item in content.get("data", [])
    )

    assert ingested_found

    # Delete the created temp file
    ingest_helper.delete_file(collection, "random_metadata")


def test_delete(test_client: TestClient, ingest_helper: IngestHelper) -> None:
    collection = str(uuid.uuid4())
    # Ingest a file
    test_client.post(
        "/v1/artifacts/ingest",
        json={
            "artifact": "file_to_keep",
            "metadata": {},
            "input": {
                "type": "text",
                "value": "text",
            },
            "collection": collection,
        },
    )

    # Ingest a second file
    test_client.post(
        "/v1/artifacts/ingest",
        json={
            "artifact": "file_to_delete",
            "metadata": {},
            "input": {
                "type": "text",
                "value": "text",
            },
            "collection": collection,
        },
    )

    response_before = test_client.get(f"/v1/artifacts/list?collection={collection}")
    ingested_before = len(response_before.json()["data"])
    assert ingested_before >= 2

    # Delete second file
    ingest_helper.delete_file(collection, "file_to_delete")

    response_after = test_client.get(f"/v1/artifacts/list?collection={collection}")
    ingested_after = len(response_after.json()["data"])
    assert ingested_after == ingested_before - 1

    # Delete the created temp file
    ingest_helper.delete_file(collection, "file_to_keep")


def test_delete_async(
    test_client: TestClient, injector: MockInjector, ingest_helper: IngestHelper
) -> None:
    collection = str(uuid.uuid4())
    _use_celery_ingestion(injector=injector)
    # Mock broker to receive callback
    broker_mock = Mock(BrokerComponent)
    injector.bind_mock(BrokerComponent, broker_mock)
    set_global_injector(injector.test_injector)

    # Ingest a file
    test_client.post(
        "/v1/artifacts/ingest",
        json={
            "artifact": "file_to_delete",
            "metadata": {},
            "input": {
                "type": "text",
                "value": "text",
            },
            "collection": collection,
        },
    )
    body = DeleteIngestedDocumentAsyncBody(
        delete_body=DeleteIngestedDocumentBody(
            collection=collection,
            artifact="file_to_delete",
        ),
        callback=Callback(
            amqp=AMQP(
                exchange="main",
                routing_key_done="delete.done",
                routing_key_progress="delete.progress",
                routing_key_error="delete.error",
            ),
            properties={"test": "123"},
        ),
    )

    response = test_client.post("/v1/artifacts/delete/async", json=body.model_dump())

    assert response.status_code == 200
    content = response.json()
    task_id = content["task_id"]
    assert task_id

    response = test_client.get(f"/v1/artifacts/delete/async/{task_id}")
    # Response will already contain the result as we are running tests synchronously
    assert response.status_code == 200
    content = response.json()
    assert content["task_id"] == task_id
    assert content["task_status"] == "SUCCESS"

    # Check if broker was called with the callback
    expected_response = AsyncResponse(
        data=None,
        error=None,
        type="pgpt.delete_ingested_task.done",
        callback_properties={"test": "123"},
    )

    broker_mock.publish.assert_called_with(
        exchange="main",
        routing_key="delete.done",
        body=bytes(expected_response.model_dump_json(), "utf-8"),
    )
