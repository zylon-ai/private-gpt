import base64
from unittest.mock import Mock, patch

import pytest

from private_gpt.artifact_index.base_artifact_index import IndexNotReadyException
from private_gpt.celery.tasks.ingestion import delete_ingested_task, vector_index_task
from private_gpt.components.storage.s3_helper import S3Helper
from private_gpt.server.ingest.ingest_router import (
    DeleteIngestedDocumentAsyncBody,
    DeleteIngestedDocumentBody,
    IngestAsyncBody,
    IngestBody,
)
from private_gpt.server.utils.artifact_input import FileArtifact, UriArtifact
from private_gpt.settings.settings import settings


@pytest.fixture
def mock_setup():
    mock_celery = Mock()
    mock_service = Mock()
    mock_injector = Mock()
    mock_injector.get.return_value = mock_service

    return mock_celery, mock_service, mock_injector


@pytest.fixture
def test_bodies():
    ingestion_body = IngestAsyncBody(
        ingest_body=IngestBody(
            input=UriArtifact(value="test_uri"),
            collection="test_collection",
            artifact="test_artifact",
        )
    )

    delete_body = DeleteIngestedDocumentAsyncBody(
        delete_body=DeleteIngestedDocumentBody(
            collection="test_collection", artifact="test_artifact"
        )
    )

    return ingestion_body, delete_body


def test_delete_already_ingested_document(mock_setup, test_bodies):
    mock_celery, mock_service, mock_injector = mock_setup
    _, delete_body = test_bodies

    mock_service.delete.return_value = None
    mock_celery.control.inspect().reserved.return_value = {}
    mock_celery.control.inspect().active.return_value = {}

    with (
        patch(
            "private_gpt.celery.tasks.ingestion.delete_tasks.celery_app", mock_celery
        ),
        patch(
            "private_gpt.celery.tasks.ingestion.delete_tasks.get_global_injector",
            return_value=mock_injector,
        ),
    ):
        delete_ingested_task(delete_body)

    mock_service.delete.assert_called_once()
    mock_celery.control.revoke.assert_not_called()


def test_delete_during_ingestion(mock_setup, test_bodies):
    mock_celery, mock_service, mock_injector = mock_setup
    ingestion_body, delete_body = test_bodies

    mock_service.delete.side_effect = [IndexNotReadyException(), None]
    mock_inspector = Mock()
    mock_inspector.reserved.return_value = {}
    mock_inspector.active.return_value = {
        "worker1": [
            {
                "id": "task1",
                "name": "vector_index_task",
                "args": [ingestion_body],
            }
        ]
    }
    mock_celery.control.inspect.return_value = mock_inspector

    with (
        patch(
            "private_gpt.celery.tasks.ingestion.delete_tasks.celery_app", mock_celery
        ),
        patch(
            "private_gpt.celery.tasks.ingestion.delete_tasks.get_global_injector",
            return_value=mock_injector,
        ),
    ):
        delete_ingested_task(delete_body)

        mock_celery.control.revoke.assert_called_once_with("task1", terminate=True)
        delete_ingested_task(delete_body)

    assert mock_service.delete.call_count == 2


def test_delete_not_started_document(mock_setup, test_bodies):
    mock_celery, mock_service, mock_injector = mock_setup
    _, delete_body = test_bodies

    mock_service.delete.side_effect = ValueError("Not initialized")
    mock_celery.control.inspect().reserved.return_value = {}
    mock_celery.control.inspect().active.return_value = {}

    with (
        patch(
            "private_gpt.celery.tasks.ingestion.delete_tasks.celery_app", mock_celery
        ),
        patch(
            "private_gpt.celery.tasks.ingestion.delete_tasks.get_global_injector",
            return_value=mock_injector,
        ),
        pytest.raises(ValueError),
    ):
        delete_ingested_task(delete_body)

    mock_service.delete.assert_called_once()
    mock_celery.control.revoke.assert_not_called()


def test_delete_different_artifact_ingesting(mock_setup, test_bodies):
    mock_celery, mock_service, mock_injector = mock_setup
    _, delete_body = test_bodies

    different_ingestion_body = IngestAsyncBody(
        ingest_body=IngestBody(
            input=UriArtifact(value="different_uri"),
            collection="test_collection",
            artifact="different_artifact",
        )
    )

    mock_service.delete.return_value = None
    mock_inspector = Mock()
    mock_inspector.reserved.return_value = {}
    mock_inspector.active.return_value = {
        "worker1": [
            {
                "id": "task1",
                "name": "vector_index_task",
                "args": [different_ingestion_body],
            }
        ]
    }
    mock_celery.control.inspect.return_value = mock_inspector

    with (
        patch(
            "private_gpt.celery.tasks.ingestion.delete_tasks.celery_app", mock_celery
        ),
        patch(
            "private_gpt.celery.tasks.ingestion.delete_tasks.get_global_injector",
            return_value=mock_injector,
        ),
    ):
        delete_ingested_task(delete_body)

    mock_service.delete.assert_called_once()
    mock_celery.control.revoke.assert_not_called()


def test_delete_with_multiple_ingestion_tasks(mock_setup, test_bodies):
    mock_celery, mock_service, mock_injector = mock_setup
    ingestion_body, delete_body = test_bodies

    same_artifact_diff_collection = IngestAsyncBody(
        ingest_body=IngestBody(
            input=UriArtifact(value="different_uri"),
            collection="different_collection",
            artifact="test_artifact",
        )
    )

    different_artifact = IngestAsyncBody(
        ingest_body=IngestBody(
            input=UriArtifact(value="different_uri"),
            collection="test_collection",
            artifact="different_artifact",
        )
    )

    mock_service.delete.side_effect = [IndexNotReadyException(), None]
    mock_inspector = Mock()
    mock_inspector.reserved.return_value = {}
    mock_inspector.active.return_value = {
        "worker1": [
            {
                "id": "task1",
                "name": "vector_index_task",
                "args": [different_artifact],
            },
            {
                "id": "task2",
                "name": "vector_index_task",
                "args": [same_artifact_diff_collection],
            },
            {
                "id": "task3",
                "name": "vector_index_task",
                "args": [ingestion_body],
            },
            {"id": "task4", "name": "different_task", "args": [{}]},
        ]
    }
    mock_celery.control.inspect.return_value = mock_inspector

    with (
        patch(
            "private_gpt.celery.tasks.ingestion.delete_tasks.celery_app", mock_celery
        ),
        patch(
            "private_gpt.celery.tasks.ingestion.delete_tasks.get_global_injector",
            return_value=mock_injector,
        ),
    ):
        delete_ingested_task(delete_body)

        mock_celery.control.revoke.assert_called_once_with("task3", terminate=True)
        assert mock_celery.control.revoke.call_count == 1


def test_delete_terminates_pending_tasks(mock_setup, test_bodies):
    mock_celery, mock_service, mock_injector = mock_setup
    ingestion_body, delete_body = test_bodies

    pending_task = {
        "id": "task1",
        "name": "vector_index_task",
        "args": [ingestion_body],
        "status": "PENDING",
    }

    mock_service.delete.side_effect = ValueError("Not initialized")
    mock_inspector = Mock()
    mock_inspector.active.return_value = {}
    mock_inspector.reserved.return_value = {"worker1": [pending_task]}
    mock_celery.control.inspect.return_value = mock_inspector

    with (
        patch(
            "private_gpt.celery.tasks.ingestion.delete_tasks.celery_app", mock_celery
        ),
        patch(
            "private_gpt.celery.tasks.ingestion.delete_tasks.get_global_injector",
            return_value=mock_injector,
        ),
    ):
        delete_ingested_task(delete_body)

        mock_celery.control.revoke.assert_called_once_with("task1", terminate=True)
        mock_service.delete.assert_called_once()


def test_delete_scheduled_when_ingestion_will_run(mock_setup, test_bodies):
    mock_celery, mock_service, mock_injector = mock_setup
    ingestion_body, delete_body = test_bodies

    mock_service.delete.side_effect = [IndexNotReadyException(), None]
    mock_inspector = Mock()
    mock_inspector.reserved.return_value = {
        "worker1": [
            {
                "id": "task1",
                "name": "delete_ingested_task",
                "args": [delete_body],
            }
        ]
    }
    mock_inspector.active.return_value = {}
    mock_celery.control.inspect.return_value = mock_inspector

    with (
        patch(
            "private_gpt.celery.tasks.ingestion.extraction_tasks.celery_app",
            mock_celery,
        ),
        patch(
            "private_gpt.di.get_global_injector",
            return_value=mock_injector,
        ),
    ):
        result = vector_index_task(ingestion_body)

        assert result.data == []
        mock_service.delete.assert_not_called()


def test_cleanup_removes_temporary_s3_file():
    temporal_bucket = settings().s3.temporary_bucket_name
    body = IngestAsyncBody(
        ingest_body=IngestBody(
            input=UriArtifact(value=f"s3://{temporal_bucket}/path/to/file.pdf"),
            collection="test_collection",
            artifact="test_artifact",
        )
    )

    temporal_bucket = settings().s3.temporary_bucket_name
    real_settings = settings()

    mock_s3_helper = Mock(spec=S3Helper)
    mock_ingest_service = Mock()

    mock_injector = Mock()

    def injector_get(cls):
        if cls == S3Helper:
            return mock_s3_helper
        elif cls.__name__ == "IngestService":
            return mock_ingest_service
        else:
            return real_settings

    mock_injector.get.side_effect = injector_get

    with patch(
        "private_gpt.di.get_global_injector",
        return_value=mock_injector,
    ):
        from private_gpt.celery.tasks.ingestion.extraction_tasks import (
            ensure_to_remove_temporal_files,
        )

        ensure_to_remove_temporal_files(body)
        mock_s3_helper.remove_file_from_s3.assert_called_once_with(
            f"s3://{temporal_bucket}/path/to/file.pdf"
        )


def test_cleanup_remove_temporary_with_failed_s3_file(mock_setup):
    mock_celery, _, _ = mock_setup

    temporal_bucket = settings().s3.temporary_bucket_name
    real_settings = settings()

    mock_s3_helper = Mock(spec=S3Helper)
    mock_ingest_service = Mock()

    mock_injector = Mock()

    def injector_get(cls):
        if cls == S3Helper:
            return mock_s3_helper
        elif cls.__name__ == "IngestService":
            return mock_ingest_service
        else:
            return real_settings

    mock_injector.get.side_effect = injector_get

    mock_celery.control.inspect().reserved.return_value = {}
    mock_celery.control.inspect().active.return_value = {}

    ingestion_body = IngestAsyncBody(
        ingest_body=IngestBody(
            input=UriArtifact(value=f"s3://{temporal_bucket}/path/to/file.pdf"),
            collection="test_collection",
            artifact="test_artifact",
        )
    )

    with (
        patch(
            "private_gpt.celery.tasks.ingestion.extraction_tasks.celery_app",
            mock_celery,
        ),
        patch(
            "private_gpt.celery.task_helper.IngestionTaskHelper.is_ingestion_cancel_task_scheduled",
            side_effect=ValueError("Ingestion failed"),
        ),
        patch(
            "private_gpt.di.get_global_injector",
            return_value=mock_injector,
        ),
    ):
        from private_gpt.celery.tasks.ingestion.extraction_tasks import (
            vector_index_task,
        )

        with pytest.raises(ValueError):
            vector_index_task(ingestion_body)

        mock_s3_helper.remove_file_from_s3.assert_called_once_with(
            f"s3://{temporal_bucket}/path/to/file.pdf"
        )


def test_cleanup_remove_temporary_with_an_autoretry_error(mock_setup):
    mock_celery, _, _ = mock_setup

    temporal_bucket = settings().s3.temporary_bucket_name
    real_settings = settings()

    mock_s3_helper = Mock(spec=S3Helper)
    mock_ingest_service = Mock()

    mock_injector = Mock()

    def injector_get(cls):
        if cls == S3Helper:
            return mock_s3_helper
        elif cls.__name__ == "IngestService":
            return mock_ingest_service
        else:
            return real_settings

    mock_injector.get.side_effect = injector_get

    mock_celery.control.inspect().reserved.return_value = {}
    mock_celery.control.inspect().active.return_value = {}

    ingestion_body = IngestAsyncBody(
        ingest_body=IngestBody(
            input=UriArtifact(value=f"s3://{temporal_bucket}/path/to/file.pdf"),
            collection="test_collection",
            artifact="test_artifact",
        )
    )

    with (
        patch(
            "private_gpt.celery.tasks.ingestion.extraction_tasks.celery_app",
            mock_celery,
        ),
        patch(
            "private_gpt.celery.task_helper.IngestionTaskHelper.is_ingestion_cancel_task_scheduled",
            side_effect=IndexNotReadyException(),
        ),
        patch(
            "private_gpt.di.get_global_injector",
            return_value=mock_injector,
        ),
    ):
        from private_gpt.celery.tasks.ingestion.extraction_tasks import (
            vector_index_task,
        )

        with pytest.raises(IndexNotReadyException):
            vector_index_task(ingestion_body)

        mock_s3_helper.remove_file_from_s3.assert_not_called()


def test_cleanup_ignores_permanent_s3_file():
    body = IngestAsyncBody(
        ingest_body=IngestBody(
            input=UriArtifact(value="s3://permanent-bucket/path/to/file.pdf"),
            collection="test_collection",
            artifact="test_artifact",
        )
    )

    mock_s3_helper = Mock(spec=S3Helper)
    mock_injector = Mock()
    mock_injector.get.return_value = mock_s3_helper

    with patch(
        "private_gpt.di.get_global_injector",
        return_value=mock_injector,
    ):
        from private_gpt.celery.tasks.ingestion.extraction_tasks import (
            ensure_to_remove_temporal_files,
        )

        ensure_to_remove_temporal_files(body)
        mock_s3_helper.remove_file_from_s3.assert_not_called()


def test_cleanup_ignores_http_uri():
    body = IngestAsyncBody(
        ingest_body=IngestBody(
            input=UriArtifact(value="https://example.com/file.pdf"),
            collection="test_collection",
            artifact="test_artifact",
        )
    )

    mock_s3_helper = Mock(spec=S3Helper)
    mock_injector = Mock()
    mock_injector.get.return_value = mock_s3_helper

    with patch(
        "private_gpt.di.get_global_injector",
        return_value=mock_injector,
    ):
        from private_gpt.celery.tasks.ingestion.extraction_tasks import (
            ensure_to_remove_temporal_files,
        )

        ensure_to_remove_temporal_files(body)
        mock_s3_helper.remove_file_from_s3.assert_not_called()


def test_cleanup_ignores_file_artifact():
    text_content = "dummy"
    base_64_content = base64.b64encode(text_content.encode("utf-8")).decode("utf-8")

    body = IngestAsyncBody(
        ingest_body=IngestBody(
            input=FileArtifact(value=base_64_content),
            collection="test_collection",
            artifact="test_artifact",
        )
    )

    mock_s3_helper = Mock(spec=S3Helper)
    mock_injector = Mock()
    mock_injector.get.return_value = mock_s3_helper

    with patch(
        "private_gpt.di.get_global_injector",
        return_value=mock_injector,
    ):
        from private_gpt.celery.tasks.ingestion.extraction_tasks import (
            ensure_to_remove_temporal_files,
        )

        ensure_to_remove_temporal_files(body)
        mock_s3_helper.remove_file_from_s3.assert_not_called()
