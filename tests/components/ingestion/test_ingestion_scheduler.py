from typing import Any
from unittest.mock import MagicMock

import pytest

from private_gpt.components.ingestion.ingestion_scheduler import (
    CeleryIngestionScheduler,
)
from private_gpt.server.ingest.ingest_router import IngestBody
from private_gpt.server.utils.artifact_input import TextArtifact, UriArtifact


@pytest.mark.parametrize(
    ("metadata", "expected_filename"),
    [
        (None, "text_content.txt"),
        ({"file_name": "sámple_¡™£¢∞§.txt"}, "sámple_¡™£¢∞§.txt"),
    ],
)
def test_sync_celery_ingest_preserves_filename_and_extension(
    monkeypatch: Any,
    metadata: dict[str, Any] | None,
    expected_filename: str,
) -> None:
    dispatched_body = None

    def dispatch_task(**kwargs: Any) -> MagicMock:
        nonlocal dispatched_body
        dispatched_body = kwargs["args"][0]
        mock = MagicMock()
        mock.task_id = "task-id"
        mock.ready.return_value = True
        mock.failed.return_value = False
        mock.result = "store-task-id"
        return mock

    monkeypatch.setattr("private_gpt.celery.dispatch.dispatch_task", dispatch_task)

    # ingest() also polls the store_vectors AsyncResult; stub it out.
    store_result_mock = MagicMock()
    store_result_mock.ready.return_value = True
    store_result_mock.failed.return_value = False
    from private_gpt.server.ingest.ingest_router import IngestResponse

    store_result_mock.result = IngestResponse(
        object="list", model="private-gpt", data=[]
    )
    monkeypatch.setattr(
        "celery.result.AsyncResult",
        lambda *a, **kw: store_result_mock,
    )

    ingest_service = MagicMock()
    s3_helper = MagicMock()
    s3_helper.is_available.return_value = True
    s3_helper.upload_file_to_s3.return_value = "s3://temporary-bucket/object-id"
    scheduler = CeleryIngestionScheduler(MagicMock(), s3_helper, ingest_service)

    scheduler.ingest(
        IngestBody(
            artifact="artifact",
            collection="collection",
            input=TextArtifact(value="sample text"),
            metadata=metadata,
        )
    )

    assert dispatched_body is not None
    assert dispatched_body.ingest_body.metadata == {"file_name": expected_filename}
    assert isinstance(dispatched_body.ingest_body.input, UriArtifact)
    assert s3_helper.upload_file_to_s3.call_args.kwargs["filename"] == expected_filename
