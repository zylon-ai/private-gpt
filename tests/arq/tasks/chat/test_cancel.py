from unittest.mock import AsyncMock

import pytest

from private_gpt.arq.tasks.chat import abort_chat_job


@pytest.mark.anyio
async def test_abort_chat_job_cancels_all_checkpoint_jobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    abort_job = AsyncMock(return_value=True)
    monkeypatch.setattr("private_gpt.arq.tasks.chat.abort_job", abort_job)

    cancelled = await abort_chat_job(
        correlation_id="chat-1",
        checkpoint_id="checkpoint-2",
        tool_ids=("tool-1", "tool-2"),
    )

    assert cancelled is True
    assert {call.kwargs["job_id"] for call in abort_job.await_args_list} == {
        "chat-1:start",
        "chat-1:resume:checkpoint-2",
        "chat-1:tool-timeout:checkpoint-2:tool-1",
        "chat-1:tool-timeout:checkpoint-2:tool-2",
        "chat-1:tool-result:tool-1",
        "chat-1:tool-result:tool-2",
    }
    assert len({call.kwargs["queue_name"] for call in abort_job.await_args_list}) == 1


@pytest.mark.anyio
async def test_abort_chat_job_waits_for_all_jobs_when_one_abort_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    abort_job = AsyncMock(
        side_effect=[RuntimeError("Redis unavailable"), True, False, True]
    )
    monkeypatch.setattr("private_gpt.arq.tasks.chat.abort_job", abort_job)

    cancelled = await abort_chat_job(
        correlation_id="chat-2",
        checkpoint_id="checkpoint-3",
        tool_ids=("tool-1",),
    )

    assert cancelled is True
    assert abort_job.await_count == 4
