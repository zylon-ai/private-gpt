import asyncio
from typing import TYPE_CHECKING, Any

from private_gpt.arq.enqueue import abort_job
from private_gpt.arq.tasks.chat.settings import get_queue_name
from private_gpt.settings.settings import settings

if TYPE_CHECKING:
    from private_gpt.arq.tasks.chat.resume import (
        enqueue_chat_timeout_job,
        enqueue_resume_iteration_job,
        enqueue_tool_resume_job,
    )
    from private_gpt.arq.tasks.chat.start import enqueue_start_chat_job


async def abort_chat_job(*, correlation_id: str) -> bool:
    results = await asyncio.gather(
        abort_job(
            job_id=f"{correlation_id}:start",
            queue_name=get_queue_name(settings()),
        ),
        abort_job(
            job_id=f"{correlation_id}:resume",
            queue_name=get_queue_name(settings()),
        ),
    )
    return any(results)


def __getattr__(name: str) -> Any:
    if name == "enqueue_start_chat_job":
        from private_gpt.arq.tasks.chat.start import enqueue_start_chat_job

        return enqueue_start_chat_job
    if name in {
        "enqueue_chat_timeout_job",
        "enqueue_resume_iteration_job",
        "enqueue_tool_resume_job",
    }:
        from private_gpt.arq.tasks.chat import resume

        return getattr(resume, name)
    raise AttributeError(name)


__all__ = [
    "abort_chat_job",
    "enqueue_chat_timeout_job",
    "enqueue_resume_iteration_job",
    "enqueue_start_chat_job",
    "enqueue_tool_resume_job",
]
