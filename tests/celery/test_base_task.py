from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from celery import Celery
from celery.backends.redis import RedisBackend

from private_gpt.celery.base import (
    MaxFailureRetriesExceeded,
    _BackgroundTask,
)


@pytest.fixture
def redis_client() -> MagicMock:
    """Create a mock Redis client for testing."""
    with patch("celery.current_app") as mock_app:
        mock_client = MagicMock()
        mock_backend = MagicMock(spec=RedisBackend)
        mock_backend.client = mock_client
        mock_app.backend = mock_backend
        yield mock_client


@pytest.fixture
def app() -> Celery:
    """Create a test Celery app with a test task."""
    celery_app = Celery(__name__)
    celery_app.conf.update(
        task_always_eager=True,
        task_eager_propagates=True,
    )

    @celery_app.task(base=_BackgroundTask, after_return=None, bind=True)
    def test_task(self: _BackgroundTask, *args: Any, **kwargs: Any) -> str:
        if kwargs.get("fail", False):
            raise ValueError("Task failed")
        return "success"

    celery_app.tasks["test_task"] = test_task
    return celery_app


def test_background_task_max_retries_exceeded(
    app: Celery, redis_client: MagicMock
) -> None:
    task = app.tasks["test_task"]

    # Mock the retry tracker
    retry_tracker = MagicMock()
    retry_tracker.increment.return_value = 4  # Exceed max_failure_retries (3)

    with patch.object(task, "_retry_tracker", retry_tracker):
        with pytest.raises(MaxFailureRetriesExceeded) as exc_info:
            task.apply_async().get()

        assert "Maximum failure retries" in str(exc_info.value)
        retry_tracker.cleanup.assert_called_once()


def test_background_task_successful_execution(
    app: Celery, redis_client: MagicMock
) -> None:
    task = app.tasks["test_task"]

    # Mock the retry tracker
    retry_tracker = MagicMock()
    retry_tracker.increment.return_value = 1

    with patch.object(task, "_retry_tracker", retry_tracker):
        result = task.apply_async().get()

        assert result == "success"
        retry_tracker.cleanup.assert_called_once()


def test_background_task_controlled_retry(app: Celery, redis_client: MagicMock) -> None:
    task = app.tasks["test_task"]

    # Mock the retry tracker
    retry_tracker = MagicMock()

    # Create a mock request object
    mock_request = MagicMock()
    mock_request.is_retry = True
    mock_request.retries = 1
    mock_request.id = "test_id"

    with (
        patch.object(task, "_retry_tracker", retry_tracker),
        patch(
            "celery.app.task.Task.request", new_callable=PropertyMock
        ) as mock_task_request,
    ):
        mock_task_request.return_value = mock_request

        result = task.apply_async().get()

        assert result == "success"
        # Since this is a controlled retry, cleanup shouldn't be called yet
        retry_tracker.cleanup.assert_called_once()


def test_background_task_failure(app: Celery, redis_client: MagicMock) -> None:
    task = app.tasks["test_task"]

    # Mock the retry tracker
    retry_tracker = MagicMock()
    retry_tracker.increment.return_value = 1

    with patch.object(task, "_retry_tracker", retry_tracker):
        with pytest.raises(ValueError) as exc_info:
            task.apply_async(kwargs={"fail": True}).get()

        assert "Task failed" in str(exc_info.value)
        retry_tracker.cleanup.assert_called_once()


def test_redis_failure_tracker_with_no_redis(app: Celery) -> None:
    task = app.tasks["test_task"]

    # Mock the celery settings to disable Redis
    with patch("private_gpt.celery.base.celery_settings") as mock_settings:
        mock_settings.backend_mode = "memory"
        mock_settings.acks_late = True

        result = task.apply_async().get()
        assert result == "success"
