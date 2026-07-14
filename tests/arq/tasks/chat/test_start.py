from unittest.mock import AsyncMock

import pytest

from private_gpt.arq.settings import START_CHAT_TASK_NAME
from private_gpt.arq.tasks.chat import enqueue_start_chat_job


@pytest.mark.parametrize(
    ("job_id", "expected_job_id"),
    [
        (None, "execution-id:start"),
        ("custom-job-id", "custom-job-id"),
    ],
)
async def test_enqueue_start_chat_job_dispatches_generic_arq_job(
    monkeypatch: pytest.MonkeyPatch,
    job_id: str | None,
    expected_job_id: str,
) -> None:
    enqueue_job = AsyncMock()
    monkeypatch.setattr(
        "private_gpt.arq.tasks.chat.start.enqueue_job",
        enqueue_job,
    )

    request_data = {"messages": [{"role": "user", "content": "Hello"}]}
    metadata = {"conversation_id": "conversation-id"}

    await enqueue_start_chat_job(
        request_data=request_data,
        correlation_id="execution-id",
        stream_type="text/event-stream",
        metadata=metadata,
        job_id=job_id,
    )

    enqueue_job.assert_awaited_once_with(
        task_name=START_CHAT_TASK_NAME,
        args=(
            request_data,
            "execution-id",
            "text/event-stream",
            metadata,
        ),
        job_id=expected_job_id,
        correlation_id="execution-id",
    )
