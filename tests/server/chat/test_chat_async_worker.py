from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from private_gpt.chat.input_models import MessageInput
from private_gpt.components.streaming.providers.in_memory_stream_service import (
    InMemoryStreamService,
)
from private_gpt.components.streaming.providers.models import StreamStatus
from private_gpt.server.chat.chat_models import ChatBody
from private_gpt.server.chat_async import (
    chat_async_service as chat_async_service_module,
)
from private_gpt.server.chat_async.chat_async_service import ChatAsyncService


class _StreamManagerStub:
    def __init__(self, stream_service: InMemoryStreamService) -> None:
        self.stream_service = stream_service
        self.cancel_stream = AsyncMock(return_value=False)


def _service(stream_service: InMemoryStreamService) -> ChatAsyncService:
    return ChatAsyncService(
        chat_facade=AsyncMock(),
        stream_manager=_StreamManagerStub(stream_service),  # type: ignore[arg-type]
        chat_request_mapper=AsyncMock(),
    )


def _chat_body() -> ChatBody:
    return ChatBody(messages=[MessageInput(role="user", content="hello")])


@pytest.mark.anyio
async def test_worker_handoff_sends_json_safe_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_service = InMemoryStreamService()
    sent: dict[str, Any] = {}

    def send_task(name: str, args: list[Any], **kwargs: Any) -> None:
        sent["name"] = name
        sent["args"] = args
        sent["kwargs"] = kwargs

    from private_gpt.celery.celery import celery_app

    monkeypatch.setattr(celery_app, "send_task", send_task)

    correlation_id = await _service(stream_service)._initiate_chat_stream_worker(
        body=_chat_body(),
        message_id="msg-1",
    )

    assert correlation_id == "msg-1"
    assert sent["name"] == "private_gpt.chat.run"
    assert sent["kwargs"]["queue"] == "chat"
    assert sent["kwargs"]["task_id"] == "msg-1"
    assert sent["args"][0] == _chat_body()
    assert sent["args"][1] == "msg-1"


@pytest.mark.anyio
async def test_worker_handoff_enqueue_failure_marks_stream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_service = InMemoryStreamService()

    def send_task(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("broker unavailable")

    from private_gpt.celery.celery import celery_app

    monkeypatch.setattr(celery_app, "send_task", send_task)

    with pytest.raises(RuntimeError, match="broker unavailable"):
        await _service(stream_service)._initiate_chat_stream_worker(
            body=_chat_body(),
            message_id="msg-2",
        )

    metadata = await stream_service.get_stream_metadata("msg-2")
    assert metadata is not None
    assert metadata.status == StreamStatus.ERROR
    assert metadata.error_message == "broker unavailable"


@pytest.mark.anyio
async def test_worker_cancel_marks_stream_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_service = InMemoryStreamService()
    await stream_service.create_stream("chat_completion", correlation_id="msg-3")
    revoked: list[str] = []

    monkeypatch.setattr(
        chat_async_service_module,
        "_settings",
        lambda: SimpleNamespace(
            scheduler=SimpleNamespace(chat=SimpleNamespace(mode="celery"))
        ),
    )
    monkeypatch.setattr(
        "private_gpt.celery.task_helper.revoke_task",
        lambda celery_app, task_id: revoked.append(task_id),
    )

    await _service(stream_service).cancel_stream("msg-3")

    metadata = await stream_service.get_stream_metadata("msg-3")
    assert metadata is not None
    assert metadata.status == StreamStatus.CANCELLED
    assert revoked == ["msg-3"]


@pytest.mark.anyio
async def test_arq_handoff_enqueues_chat_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_service = InMemoryStreamService()
    sent: dict[str, Any] = {}

    async def fake_enqueue_chat_job(**kwargs: Any) -> None:
        sent.update(kwargs)

    monkeypatch.setattr(
        chat_async_service_module,
        "enqueue_chat_job",
        fake_enqueue_chat_job,
    )

    correlation_id = await _service(stream_service)._initiate_chat_stream_arq(
        body=_chat_body(),
        message_id="msg-arq-1",
    )

    assert correlation_id == "msg-arq-1"
    assert sent["correlation_id"] == "msg-arq-1"
    assert sent["stream_type"] == "chat_completion"
    assert sent["body"] == _chat_body().model_dump(mode="json")


@pytest.mark.anyio
async def test_arq_cancel_marks_stream_cancelled_without_revoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_service = InMemoryStreamService()
    await stream_service.create_stream("chat_completion", correlation_id="msg-4")
    revoke_calls: list[str] = []

    monkeypatch.setattr(
        chat_async_service_module,
        "_settings",
        lambda: SimpleNamespace(
            scheduler=SimpleNamespace(chat=SimpleNamespace(mode="arq"))
        ),
    )
    monkeypatch.setattr(
        "private_gpt.celery.task_helper.revoke_task",
        lambda celery_app, task_id: revoke_calls.append(task_id),
    )

    await _service(stream_service).cancel_stream("msg-4")

    metadata = await stream_service.get_stream_metadata("msg-4")
    assert metadata is not None
    assert metadata.status == StreamStatus.CANCELLED
    assert revoke_calls == []
