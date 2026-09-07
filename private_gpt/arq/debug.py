"""Runtime dumps for a wedged ARQ worker.

``/health`` is a separate uvicorn process. It only GETs the Redis sentinel, so
it stays 200 while a job is stuck in ``await`` and the loop is still ticking.

Dump from the worker process (PID logged at startup; in the container it is
usually PID 1, not the uvicorn child):

- ``kill -USR1 <pid>`` — asyncio tasks, ARQ jobs, Python thread stacks
- ``kill -USR2 <pid>`` — C/native stacks (use if the loop is blocked in gRPC)
- ``GET :9464/metrics`` — Prometheus gauges (occupancy, job age, loop lag)
- ``GET :9464/debug`` — JSON snapshot + stacks (works even if the loop is stuck)

These dumps only log. They do not change ``/health``.
"""

from __future__ import annotations

import asyncio
import contextlib
import faulthandler
import io
import logging
import os
import signal
import sys
import threading
import time
import traceback
from typing import Any

logger = logging.getLogger(__name__)

_DUMP_REQUESTED = threading.Event()
_WORKER: Any = None
_LOOP: asyncio.AbstractEventLoop | None = None
_JOB_FIRST_SEEN: dict[str, float] = {}
_JOB_LAST_DUMP: dict[str, float] = {}
_SIGNALS_INSTALLED = False
_SNAPSHOT_LOCK = threading.Lock()
_SNAPSHOT: dict[str, Any] = {
    "event_loop_lag_seconds": 0.0,
    "event_loop_heartbeat_unix": 0.0,
    "asyncio_tasks": 0,
}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _write(msg: str) -> None:
    data = msg if msg.endswith("\n") else msg + "\n"
    with contextlib.suppress(OSError):
        os.write(2, data.encode("utf-8", "replace"))


def _task_status(task: asyncio.Task[Any]) -> str:
    if task.cancelled():
        return "cancelled"
    if task.done():
        return "done"
    return "pending"


def _format_task(task: asyncio.Task[Any]) -> str:
    lines = [f"- {task.get_name()} status={_task_status(task)} {task!s}"]
    coro = task.get_coro()
    awaited = getattr(coro, "cr_await", None) or getattr(coro, "ag_await", None)
    if awaited is not None:
        lines.append("    await " + repr(awaited)[:500])
    buf = io.StringIO()
    task.print_stack(limit=12, file=buf)
    stack = buf.getvalue().rstrip()
    if stack:
        lines.append(stack)
    else:
        lines.append("    <no python stack; waiting in C/extension or unstarted>")
    return "\n".join(lines)


def _format_threads() -> str:
    frames = sys._current_frames()
    ident_to_thread = {t.ident: t for t in threading.enumerate()}
    chunks: list[str] = []
    for ident, frame in frames.items():
        thread = ident_to_thread.get(ident)
        name = thread.name if thread is not None else f"tid-{ident}"
        chunks.append(f"Thread {name} ident={ident}")
        chunks.append("".join(traceback.format_stack(frame)).rstrip())
    return "\n".join(chunks) if chunks else "<no threads>"


def _job_ids(mapping: Any) -> list[str]:
    if not mapping:
        return []
    return sorted(str(job_id) for job_id in mapping)


def format_debug_dump(
    *,
    worker: Any | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
) -> str:
    """Snapshot asyncio tasks, ARQ jobs, and thread stacks."""
    worker = worker if worker is not None else _WORKER
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"ARQ debug dump ts={now} pid={os.getpid()} "
        f"thread={threading.current_thread().name} "
        f"nest_asyncio={'nest_asyncio' in sys.modules}",
    ]
    if worker is not None:
        tasks = getattr(worker, "tasks", {}) or {}
        job_tasks = getattr(worker, "job_tasks", {}) or {}
        lines.append(
            "arq "
            f"jobs_complete={getattr(worker, 'jobs_complete', '?')} "
            f"jobs_failed={getattr(worker, 'jobs_failed', '?')} "
            f"j_ongoing={len(tasks)} "
            f"job_tasks={len(job_tasks)} "
            f"max_jobs={getattr(worker, 'max_jobs', '?')} "
            f"job_timeout_s={getattr(worker, 'job_timeout_s', '?')}"
        )
        in_flight = _job_ids(tasks)
        lines.append(
            "arq in-flight: " + (", ".join(in_flight) if in_flight else "none")
        )
        named = _job_ids(job_tasks)
        if named:
            lines.append("arq job_tasks: " + ", ".join(named))

    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = _LOOP
    if loop is not None:
        aio_tasks = asyncio.all_tasks(loop)
        lines.append(f"asyncio tasks={len(aio_tasks)}")
        for task in sorted(aio_tasks, key=lambda item: item.get_name()):
            lines.append(_format_task(task))
    else:
        lines.append("asyncio loop: not running in this thread")

    lines.append(f"threads ({threading.active_count()}):")
    lines.append(_format_threads())
    return "\n".join(lines)


def dump_now(*, worker: Any | None = None) -> str:
    snapshot = format_debug_dump(worker=worker)
    logger.warning("ARQ debug dump\n%s", snapshot)
    _write(snapshot)
    return snapshot


def _on_usr1(signum: int, frame: object | None) -> None:
    del signum
    _write("ARQ SIGUSR1: python stacks (faulthandler)")
    if frame is not None:
        _write(
            "interrupted frame:\n" + "".join(traceback.format_stack(frame, limit=8))  # ty:ignore[invalid-argument-type]
        )
    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
    try:
        _write(format_debug_dump())
    except Exception as exc:
        _write(f"ARQ SIGUSR1 asyncio dump failed: {exc!r}")
        _DUMP_REQUESTED.set()


def _install_signals() -> None:
    global _SIGNALS_INSTALLED
    if _SIGNALS_INSTALLED:
        return
    faulthandler.enable(all_threads=True)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, _on_usr1)
    if hasattr(signal, "SIGUSR2"):
        try:
            faulthandler.register(signal.SIGUSR2, file=sys.stderr, all_threads=True)
        except Exception:
            logger.warning("Could not register SIGUSR2 faulthandler", exc_info=True)
    _SIGNALS_INSTALLED = True


def stuck_job_ids(
    worker: Any,
    now: float,
    warn_s: float,
    repeat_s: float,
) -> list[str]:
    live = {str(job_id) for job_id in (getattr(worker, "tasks", {}) or {})}
    for job_id in list(_JOB_FIRST_SEEN):
        if job_id not in live:
            _JOB_FIRST_SEEN.pop(job_id, None)
            _JOB_LAST_DUMP.pop(job_id, None)
    stuck: list[str] = []
    for job_id in live:
        first = _JOB_FIRST_SEEN.setdefault(job_id, now)
        age = now - first
        if age < warn_s:
            continue
        last = _JOB_LAST_DUMP.get(job_id)
        if last is not None and now - last < repeat_s:
            continue
        _JOB_LAST_DUMP[job_id] = now
        stuck.append(f"{job_id} age={age:.0f}s")
    return stuck


def reset_debug_state() -> None:
    """Test helper."""
    _JOB_FIRST_SEEN.clear()
    _JOB_LAST_DUMP.clear()
    _DUMP_REQUESTED.clear()
    with _SNAPSHOT_LOCK:
        _SNAPSHOT.clear()
        _SNAPSHOT.update(
            {
                "event_loop_lag_seconds": 0.0,
                "event_loop_heartbeat_unix": 0.0,
                "asyncio_tasks": 0,
            }
        )


def collect_worker_stats() -> dict[str, Any]:
    """Numbers for /metrics. Safe to call from the metrics HTTP thread."""
    worker = _WORKER
    tasks = getattr(worker, "tasks", {}) or {} if worker is not None else {}
    live = {str(job_id) for job_id in tasks}
    now = time.monotonic()
    for job_id in list(_JOB_FIRST_SEEN):
        if job_id not in live:
            _JOB_FIRST_SEEN.pop(job_id, None)
            _JOB_LAST_DUMP.pop(job_id, None)
    for job_id in live:
        _JOB_FIRST_SEEN.setdefault(job_id, now)
    ages = [now - _JOB_FIRST_SEEN[job_id] for job_id in live]
    with _SNAPSHOT_LOCK:
        lag = float(_SNAPSHOT.get("event_loop_lag_seconds") or 0.0)
        heartbeat = float(_SNAPSHOT.get("event_loop_heartbeat_unix") or 0.0)
        aio_tasks = int(_SNAPSHOT.get("asyncio_tasks") or 0)
    return {
        "pid": os.getpid(),
        "jobs_ongoing": len(tasks),
        "jobs_max": int(getattr(worker, "max_jobs", 0) or 0)
        if worker is not None
        else 0,
        "jobs_complete": int(getattr(worker, "jobs_complete", 0) or 0)
        if worker is not None
        else 0,
        "jobs_failed": int(getattr(worker, "jobs_failed", 0) or 0)
        if worker is not None
        else 0,
        "job_timeout_s": float(getattr(worker, "job_timeout_s", 0) or 0)
        if worker is not None
        else 0.0,
        "oldest_job_age_seconds": max(ages, default=0.0),
        "event_loop_lag_seconds": lag,
        "event_loop_heartbeat_unix": heartbeat,
        "asyncio_tasks": aio_tasks,
        "in_flight": sorted(live),
    }


def _record_loop_sample(*, lag_seconds: float, asyncio_tasks: int) -> None:
    with _SNAPSHOT_LOCK:
        _SNAPSHOT["event_loop_lag_seconds"] = max(0.0, lag_seconds)
        _SNAPSHOT["event_loop_heartbeat_unix"] = time.time()
        _SNAPSHOT["asyncio_tasks"] = asyncio_tasks


def build_debug_payload(*, dump_timeout_s: float = 1.0) -> dict[str, Any]:
    """JSON for GET /debug. Thread stacks always; asyncio dump if the loop answers."""
    stats = collect_worker_stats()
    heartbeat = float(stats.get("event_loop_heartbeat_unix") or 0.0)
    heartbeat_age = (time.time() - heartbeat) if heartbeat else None
    payload: dict[str, Any] = {
        **stats,
        "heartbeat_age_seconds": heartbeat_age,
        "loop_blocked": bool(heartbeat_age is not None and heartbeat_age > 5.0),
        "threads": _format_threads(),
        "dump": None,
    }
    loop = _LOOP
    if loop is None or not loop.is_running():
        return payload

    async def _dump() -> str:
        return format_debug_dump()

    future = asyncio.run_coroutine_threadsafe(_dump(), loop)
    try:
        payload["dump"] = future.result(timeout=dump_timeout_s)
        payload["loop_blocked"] = False
    except TimeoutError:
        payload["loop_blocked"] = True
        future.cancel()
    except Exception as exc:
        payload["dump_error"] = repr(exc)
    return payload


async def _monitor() -> None:
    warn_ms = _env_float("PGPT_ARQ_LOOP_LAG_WARN_MS", 2000.0)
    stuck_s = _env_float("PGPT_ARQ_JOB_STUCK_WARN_S", 1800.0)
    repeat_s = _env_float("PGPT_ARQ_JOB_STUCK_REPEAT_S", 1800.0)
    interval = 1.0
    while True:
        if _DUMP_REQUESTED.is_set():
            _DUMP_REQUESTED.clear()
            try:
                dump_now()
            except Exception:
                logger.exception("ARQ debug dump failed")
        t0 = time.monotonic()
        await asyncio.sleep(interval)
        lag_s = time.monotonic() - t0 - interval
        try:
            aio_tasks = len(asyncio.all_tasks())
        except Exception:
            aio_tasks = 0
        _record_loop_sample(lag_seconds=lag_s, asyncio_tasks=aio_tasks)
        lag_ms = lag_s * 1000.0
        if lag_ms >= warn_ms:
            logger.warning(
                "event loop lag %.0fms (threshold %.0fms) — loop was blocked",
                lag_ms,
                warn_ms,
            )
            try:
                dump_now()
            except Exception:
                logger.exception("ARQ debug dump failed")
        worker = _WORKER
        if worker is not None and stuck_s > 0:
            stuck = stuck_job_ids(worker, time.monotonic(), stuck_s, repeat_s)
            if stuck:
                logger.warning("ARQ jobs running too long: %s", ", ".join(stuck))
                try:
                    dump_now(worker=worker)
                except Exception:
                    logger.exception("ARQ debug dump failed")


def start_worker_debug(worker: Any) -> asyncio.Task[None] | None:
    """Install signal dumps and a loop-lag / stuck-job monitor.

    Set ``PGPT_ARQ_DEBUG=0`` to keep signals but skip the periodic monitor.
    """
    global _WORKER, _LOOP
    _WORKER = worker
    try:
        _LOOP = asyncio.get_running_loop()
    except RuntimeError:
        _LOOP = None
    _install_signals()
    from private_gpt.arq.metrics import start_metrics_server

    metrics_port = start_metrics_server()
    logger.info(
        "ARQ debug enabled pid=%s SIGUSR1=asyncio/threads SIGUSR2=native stacks "
        "loop_lag_warn_ms=%s job_stuck_warn_s=%s metrics_port=%s",
        os.getpid(),
        _env_float("PGPT_ARQ_LOOP_LAG_WARN_MS", 2000.0),
        _env_float("PGPT_ARQ_JOB_STUCK_WARN_S", 1800.0),
        metrics_port if metrics_port is not None else "off",
    )
    disabled = os.environ.get("PGPT_ARQ_DEBUG", "1").strip().lower() in {
        "0",
        "false",
        "off",
    }
    if disabled:
        return None
    return asyncio.create_task(_monitor(), name="arq_debug_monitor")
