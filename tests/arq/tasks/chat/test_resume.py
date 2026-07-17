from unittest.mock import AsyncMock

import pytest

from private_gpt.arq.tasks.chat.resume import enqueue_tool_resume_job


@pytest.mark.anyio
async def test_enqueue_tool_resume_job_passes_error_result_as_arq_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enqueue_job = AsyncMock()
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

    await enqueue_tool_resume_job(
        correlation_id="chat-1",
        tool_id="semantic-search-1",
        result=result,
    )

    enqueue_job.assert_awaited_once()
    call = enqueue_job.await_args.kwargs
    assert call["args"] == ("chat-1", "semantic-search-1", result)
    assert call["correlation_id"] == "chat-1"
