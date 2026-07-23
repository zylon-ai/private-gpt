import asyncio
from typing import TYPE_CHECKING, Any

from private_gpt.arq.enqueue import abort_job
from private_gpt.arq.tasks.chat.settings import get_queue_name
from private_gpt.settings.settings import settings

if TYPE_CHECKING:
    from private_gpt.arq.tasks.chat.resume import (
        enqueue_resume_iteration_job,
        enqueue_tool_resume_job,
        enqueue_tool_timeout_job,
    )
    from private_gpt.arq.tasks.chat.start import enqueue_start_chat_job


async def abort_chat_job(
    *,
    correlation_id: str,
    checkpoint_id: str | None = None,
    tool_ids: tuple[str, ...] = (),
) -> bool:
    job_ids = [f"{correlation_id}:start"]
    if checkpoint_id:
        job_ids.extend(
            (
                f"{correlation_id}:resume:{checkpoint_id}",
                *(
                    f"{correlation_id}:tool-timeout:{checkpoint_id}:{tool_id}"
                    for tool_id in tool_ids
                ),
            )
        )
    job_ids.extend(f"{correlation_id}:tool-result:{tool_id}" for tool_id in tool_ids)
    results = await asyncio.gather(
        *(
            abort_job(job_id=job_id, queue_name=get_queue_name(settings()))
            for job_id in job_ids
        ),
        return_exceptions=True,
    )
    return any(result is True for result in results)


def __getattr__(name: str) -> Any:
    if name == "enqueue_start_chat_job":
        from private_gpt.arq.tasks.chat.start import enqueue_start_chat_job

        return enqueue_start_chat_job
    if name in {
        "enqueue_resume_iteration_job",
        "enqueue_tool_resume_job",
        "enqueue_tool_timeout_job",
    }:
        from private_gpt.arq.tasks.chat import resume

        return getattr(resume, name)
    raise AttributeError(name)


__all__ = [
    "abort_chat_job",
    "enqueue_resume_iteration_job",
    "enqueue_start_chat_job",
    "enqueue_tool_resume_job",
    "enqueue_tool_timeout_job",
]
