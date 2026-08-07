"""Tests for the FileCallbackService delivery worker.

The delivery worker mirrors the broker publisher: jobs are enqueued onto a
background thread that performs the blocking HTTP delivery, so webhook
handlers never block on callback I/O.  The worker is only started in the
API process (when ``PGPT_WORKER_MODE`` is unset).
"""

from __future__ import annotations

import os
import threading
from unittest.mock import patch

import pytest

from private_gpt.components.filesystems.callbacks.file_callback_service import (
    FileCallbackService,
    _CallbackWorker,
    _is_worker_process,
)
from private_gpt.components.filesystems.callbacks.file_watcher import (
    FileEventDebouncer,
)
from private_gpt.components.filesystems.callbacks.models import (
    FileCallbackTarget,
    HttpCallbackTarget,
)
from private_gpt.components.filesystems.namespace_registry import NamespaceRegistry


def _make_service() -> FileCallbackService:
    """Build a service without DI, exactly as the injector would (registry only)."""
    service = FileCallbackService.__new__(FileCallbackService)
    service._registry = NamespaceRegistry.__new__(NamespaceRegistry)
    service._sessions = {}
    service._debouncers = {}
    service._bucket_namespace_map = {}
    service._lock = threading.RLock()
    service._worker = None
    return service


def _minio_payload(key: str = "org-1/art-1.mdx", size: int = 1024) -> dict:
    return {
        "Records": [
            {
                "eventName": "s3:ObjectCreated:Put",
                "s3": {
                    "bucket": {"name": "artifacts"},
                    "object": {"key": key, "size": size},
                },
            }
        ]
    }


def _watch_session(service: FileCallbackService) -> None:
    service._debouncers["artifacts"] = FileEventDebouncer()
    service._sessions["s1"] = (
        "artifacts",
        FileCallbackTarget(http=HttpCallbackTarget(url="http://127.0.0.1:9/cb")),
    )


class _FakeResponse:
    status_code = 200


class TestIsWorkerProcess:
    def test_default_api_process(self) -> None:
        with patch.dict("os.environ", {}, clear=False):
            os.environ.pop("PGPT_WORKER_MODE", None)
            assert _is_worker_process() is False

    def test_worker_process(self) -> None:
        with patch.dict("os.environ", {"PGPT_WORKER_MODE": "celery"}, clear=False):
            assert _is_worker_process() is True


class TestCallbackDeliveryWorker:
    @pytest.mark.asyncio
    async def test_http_event_delivered_on_background_thread(self) -> None:
        service = _make_service()
        _watch_session(service)
        worker = _CallbackWorker()
        service._worker = worker
        worker.start()

        delivered: dict[str, object] = {}
        main_thread = threading.get_ident()

        def fake_post(*args: object, **kwargs: object) -> object:
            delivered["thread_is_worker"] = threading.get_ident() != main_thread
            delivered["json"] = kwargs.get("json")
            delivered["headers"] = kwargs.get("headers")
            delivered["timeout"] = kwargs.get("timeout")
            return _FakeResponse()

        with patch(
            "private_gpt.components.filesystems.callbacks.file_callback_service.requests.post",
            side_effect=fake_post,
        ):
            await service.handle_bucket_notification("artifacts", _minio_payload())
            service.drain()

        assert delivered["thread_is_worker"] is True
        assert delivered["json"]["type"] == "file.created"
        assert delivered["json"]["path"] == "art-1.mdx"
        assert delivered["json"]["namespace"] == "artifacts"
        assert delivered["json"]["scope"] == "org-1"
        assert delivered["headers"] == {"Content-Type": "application/json"}
        assert delivered["timeout"] == 5
        worker.close()

    @pytest.mark.asyncio
    async def test_enqueues_job_without_blocking_delivery(self) -> None:
        service = _make_service()
        _watch_session(service)
        worker = _CallbackWorker()
        service._worker = worker
        worker.start()

        await service.handle_bucket_notification("artifacts", _minio_payload())
        # The handler must return immediately with the job queued, not delivered.
        assert worker._queue.unfinished_tasks == 1
        worker.close()

    @pytest.mark.asyncio
    async def test_worker_process_drops_events_without_delivery(self) -> None:
        service = _make_service()
        _watch_session(service)
        # No delivery worker: simulates a worker process (PGPT_WORKER_MODE set).
        assert service._worker is None

        with patch(
            "private_gpt.components.filesystems.callbacks.file_callback_service.requests.post"
        ) as mock_post:
            await service.handle_bucket_notification("artifacts", _minio_payload())
            mock_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_drains_queued_jobs(self) -> None:
        service = _make_service()
        _watch_session(service)
        worker = _CallbackWorker()
        service._worker = worker
        worker.start()

        delivered: list[object] = []

        def fake_post(*args: object, **kwargs: object) -> object:
            delivered.append(kwargs.get("json"))
            return _FakeResponse()

        with patch(
            "private_gpt.components.filesystems.callbacks.file_callback_service.requests.post",
            side_effect=fake_post,
        ):
            await service.handle_bucket_notification("artifacts", _minio_payload())
            service.close()

        assert len(delivered) == 1
        assert not worker.is_alive()
