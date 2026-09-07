from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from private_gpt.arq.debug import (
    collect_worker_stats,
    format_debug_dump,
    reset_debug_state,
    start_worker_debug,
    stuck_job_ids,
)
from private_gpt.arq.metrics import (
    render_prometheus,
    reset_metrics_server,
    start_metrics_server,
)


@pytest.fixture(autouse=True)
def _metrics_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PGPT_ARQ_METRICS_PORT", "0")
    reset_debug_state()
    reset_metrics_server()
    yield
    reset_metrics_server()
    reset_debug_state()


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


def test_metrics_render_occupancy_and_job_age(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PGPT_ARQ_DEBUG", "0")
    worker = SimpleNamespace(
        tasks={"job-1": object()},
        jobs_complete=4,
        jobs_failed=2,
        max_jobs=1,
        job_timeout_s=21600,
    )
    start_worker_debug(worker)
    stats = collect_worker_stats()
    assert stats["jobs_ongoing"] == 1
    assert stats["jobs_max"] == 1
    assert stats["in_flight"] == ["job-1"]
    body = render_prometheus(stats)
    assert "arq_worker_jobs_ongoing 1" in body
    assert "arq_worker_jobs_max 1" in body
    assert "arq_worker_up 1" in body


def test_metrics_http_endpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from urllib.request import urlopen

    monkeypatch.setenv("PGPT_ARQ_DEBUG", "0")
    worker = SimpleNamespace(
        tasks={"job-http": object()},
        jobs_complete=0,
        jobs_failed=0,
        max_jobs=2,
        job_timeout_s=30,
    )
    start_worker_debug(worker)
    port = start_metrics_server(0)
    assert port is not None
    assert port > 0
    with urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as response:
        body = response.read().decode()
        assert response.status == 200
    assert "arq_worker_jobs_ongoing 1" in body
    with urlopen(f"http://127.0.0.1:{port}/debug", timeout=2) as response:
        payload = json.loads(response.read().decode())
        assert response.status == 200
    assert payload["jobs_ongoing"] == 1
    assert payload["in_flight"] == ["job-http"]
    assert "threads" in payload
