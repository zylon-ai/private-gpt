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


def test_celery_bytes_to_text_times_out_and_revokes_lost_task(
    monkeypatch: Any,
) -> None:
    """Regression: a parse task that never completes blocked chat preprocessing
    forever (no tool result, chat stuck until the ARQ job timeout)."""
    import threading

    from celery.exceptions import TimeoutError as CeleryTimeoutError

    never_ready = MagicMock()
    never_ready.id = "parse-task-id"
    never_ready.ready.return_value = False
    never_ready.failed.return_value = False
    monkeypatch.setattr(
        "private_gpt.celery.dispatch.dispatch_task",
        MagicMock(return_value=never_ready),
    )
    celery_app = MagicMock()
    monkeypatch.setattr("private_gpt.celery.celery.celery_app", celery_app)

    settings = MagicMock()
    settings.scheduler.ingestion.celery_queue = "ingestion"
    settings.scheduler.ingestion.convert_timeout_seconds = 1
    scheduler = CeleryIngestionScheduler(settings, MagicMock(), MagicMock())

    outcome: dict[str, BaseException] = {}

    def run() -> None:
        try:
            scheduler.bytes_to_text(b"payload", ".pptx")
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), "bytes_to_text hung waiting for a lost parse task"
    assert isinstance(outcome.get("error"), CeleryTimeoutError)
    celery_app.control.revoke.assert_called_once_with("parse-task-id", terminate=True)


@pytest.mark.parametrize(
    ("parse_ready", "lost_task_id"),
    [
        (False, "parse-task-id"),
        (True, "store-task-id"),
    ],
)
def test_celery_sync_ingest_times_out_and_revokes_lost_task(
    monkeypatch: Any, parse_ready: bool, lost_task_id: str
) -> None:
    """Regression: ``ingest()`` waited forever on the parse and store tasks, so a
    lost worker turned a sync ingest request into a hung HTTP call."""
    import threading

    from celery.exceptions import TimeoutError as CeleryTimeoutError

    parse_result = MagicMock()
    parse_result.id = "parse-task-id"
    parse_result.ready.return_value = parse_ready
    parse_result.failed.return_value = False
    parse_result.result = "store-task-id"
    monkeypatch.setattr(
        "private_gpt.celery.dispatch.dispatch_task",
        MagicMock(return_value=parse_result),
    )
    store_result = MagicMock()
    store_result.id = "store-task-id"
    store_result.ready.return_value = False
    store_result.failed.return_value = False
    monkeypatch.setattr(
        "celery.result.AsyncResult", MagicMock(return_value=store_result)
    )
    celery_app = MagicMock()
    monkeypatch.setattr("private_gpt.celery.celery.celery_app", celery_app)

    settings = MagicMock()
    settings.scheduler.ingestion.celery_queue = "ingestion"
    settings.scheduler.ingestion.ingest_timeout_seconds = 1
    s3_helper = MagicMock()
    s3_helper.is_available.return_value = False
    scheduler = CeleryIngestionScheduler(settings, s3_helper, MagicMock())

    outcome: dict[str, BaseException] = {}

    def run() -> None:
        try:
            scheduler.ingest(
                IngestBody(
                    artifact="artifact",
                    collection="collection",
                    input=TextArtifact(value="sample text"),
                )
            )
        except BaseException as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(timeout=5)
    assert not worker.is_alive(), "ingest hung waiting for a lost Celery task"
    assert isinstance(outcome.get("error"), CeleryTimeoutError)
    celery_app.control.revoke.assert_called_once_with(lost_task_id, terminate=True)
