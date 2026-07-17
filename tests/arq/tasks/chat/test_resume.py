from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.arq.tasks.chat.resume import (
    enqueue_resume_iteration_job,
    enqueue_tool_resume_job,
    enqueue_tool_timeout_job,
    timeout_tool_job,
)


@pytest.mark.anyio
async def test_enqueue_resume_iteration_job_allows_multiple_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue_job = AsyncMock(return_value=False)
    monkeypatch.setattr("private_gpt.arq.tasks.chat.resume.enqueue_job", enqueue_job)

    await enqueue_resume_iteration_job(
        correlation_id="chat-1", job_id="chat-1:resume:checkpoint-1"
    )
    await enqueue_resume_iteration_job(
        correlation_id="chat-1", job_id="chat-1:resume:checkpoint-2"
    )

    assert enqueue_job.await_count == 2
    job_ids = []
    for call in enqueue_job.await_args_list:
        assert call.kwargs["args"] == ("chat-1",)
        assert call.kwargs["correlation_id"] == "chat-1"
        assert call.kwargs["job_id"].startswith("chat-1:resume:checkpoint-")
        job_ids.append(call.kwargs["job_id"])

    assert job_ids[0] != job_ids[1]


@pytest.mark.anyio
async def test_enqueue_tool_resume_job_passes_error_result_as_arq_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue_job = AsyncMock(return_value=False)
    monkeypatch.setattr("private_gpt.arq.tasks.chat.resume.enqueue_job", enqueue_job)
    result = {
        "tool_name": "semantic_search",
        "tool_id": "semantic-search-1",
        "result_content": [{"type": "text", "text": "query: Field required"}],
        "is_error": True,
        "tool_message": {
            "role": "tool",
            "content": "query: Field required",
            "additional_kwargs": {"tool_call_id": "semantic-search-1"},
        },
    }

    accepted = await enqueue_tool_resume_job(
        correlation_id="chat-1",
        tool_id="semantic-search-1",
        result=result,
    )

    assert accepted is False
    enqueue_job.assert_awaited_once()
    call = enqueue_job.await_args.kwargs
    assert call["args"] == ("chat-1", "semantic-search-1", result)
    assert call["correlation_id"] == "chat-1"
    assert call["job_id"] == "chat-1:tool-result:semantic-search-1"


@pytest.mark.anyio
async def test_real_result_and_timeout_publish_the_same_tool_result_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue_job = AsyncMock(side_effect=[True, False])
    monkeypatch.setattr("private_gpt.arq.tasks.chat.resume.enqueue_job", enqueue_job)

    real_accepted = await enqueue_tool_resume_job(
        correlation_id="chat-1",
        tool_id="tool-1",
        result={"source": "real"},
    )
    timeout_accepted = await enqueue_tool_resume_job(
        correlation_id="chat-1",
        tool_id="tool-1",
        result={"source": "timeout"},
    )

    assert real_accepted is True
    assert timeout_accepted is False
    assert {
        call.kwargs["job_id"] for call in enqueue_job.await_args_list
    } == {"chat-1:tool-result:tool-1"}


@pytest.mark.anyio
async def test_enqueue_tool_timeout_job_is_deferred_and_separate_from_result_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue_job = AsyncMock(return_value=True)
    monkeypatch.setattr("private_gpt.arq.tasks.chat.resume.enqueue_job", enqueue_job)

    await enqueue_tool_timeout_job(
        correlation_id="chat-1",
        checkpoint_id="checkpoint-1",
        tool_id="tool-1",
        tool_name="search",
        task_id="celery-task-1",
        delay_seconds=30,
    )

    call = enqueue_job.await_args.kwargs
    assert call["job_id"] == "chat-1:tool-timeout:checkpoint-1:tool-1"
    assert call["defer_seconds"] == 30
    assert call["args"] == (
        "chat-1",
        "tool-1",
        "search",
        "celery-task-1",
        30,
    )


@pytest.mark.anyio
async def test_timeout_cancels_real_tool_only_when_timeout_result_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue_result = AsyncMock(side_effect=[True, False])
    scheduler = MagicMock()
    scheduler.cancel_task = AsyncMock(return_value=True)
    scheduler_factory = MagicMock()
    scheduler_factory.get.return_value = scheduler
    injector = MagicMock()
    injector.get.return_value = scheduler_factory
    monkeypatch.setattr(
        "private_gpt.arq.tasks.chat.resume.enqueue_tool_resume_job", enqueue_result
    )
    monkeypatch.setattr(
        "private_gpt.arq.tasks.chat.resume.get_global_injector",
        lambda **_: injector,
    )

    await timeout_tool_job(
        {}, "chat-1", "tool-1", "search", "celery-task-1", 30
    )
    await timeout_tool_job(
        {}, "chat-1", "tool-1", "search", "celery-task-1", 30
    )

    scheduler.cancel_task.assert_awaited_once_with("celery-task-1")
