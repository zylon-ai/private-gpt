from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from private_gpt.arq.debug import (
    format_debug_dump,
    reset_debug_state,
    start_worker_debug,
    stuck_job_ids,
)


async def test_format_debug_dump_includes_arq_jobs_and_asyncio_tasks() -> None:
    async def waiter() -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(waiter(), name="stuck_chat")
    try:
        worker = SimpleNamespace(
            tasks={"job-1": task},
            job_tasks={"job-1": task},
            jobs_complete=3,
            jobs_failed=1,
            max_jobs=1,
            job_timeout_s=21600,
        )
        snapshot = format_debug_dump(worker=worker)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "j_ongoing=1" in snapshot
    assert "job-1" in snapshot
    assert "stuck_chat" in snapshot
    assert "threads" in snapshot


def test_stuck_job_ids_warns_then_respects_repeat_window() -> None:
    reset_debug_state()
    worker = SimpleNamespace(tasks={"chat-1": object()})

    assert stuck_job_ids(worker, now=0.0, warn_s=10.0, repeat_s=30.0) == []
    assert stuck_job_ids(worker, now=9.0, warn_s=10.0, repeat_s=30.0) == []
    first = stuck_job_ids(worker, now=11.0, warn_s=10.0, repeat_s=30.0)
    assert first == ["chat-1 age=11s"]
    assert stuck_job_ids(worker, now=20.0, warn_s=10.0, repeat_s=30.0) == []
    again = stuck_job_ids(worker, now=41.0, warn_s=10.0, repeat_s=30.0)
    assert again == ["chat-1 age=41s"]

    worker.tasks = {}
    assert stuck_job_ids(worker, now=50.0, warn_s=10.0, repeat_s=30.0) == []
    worker.tasks = {"chat-1": object()}
    assert stuck_job_ids(worker, now=50.0, warn_s=10.0, repeat_s=30.0) == []


async def test_start_worker_debug_can_skip_monitor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PGPT_ARQ_DEBUG", "0")
    worker = SimpleNamespace(tasks={}, job_tasks={})
    assert start_worker_debug(worker) is None
