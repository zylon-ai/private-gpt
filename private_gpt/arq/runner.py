import asyncio
import contextlib
import os
import signal
import subprocess
import sys
from collections.abc import Callable

from arq.typing import StartupShutdown
from arq.worker import Worker

from private_gpt.arq.hooks import on_job_end
from private_gpt.arq.lifecycle import shutdown, startup
from private_gpt.arq.settings import get_queue_name, get_redis_settings
from private_gpt.arq.tasks import autodiscover_registered_tasks
from private_gpt.settings.settings import Settings, settings


def _default_concurrency() -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1:
        return 1
    return 1 << (cpu_count.bit_length() - 1)


def _task_packages() -> tuple[str, ...]:
    configured = os.environ.get("PGPT_ARQ_TASK_PACKAGES", "")
    task_packages = tuple(
        package.strip() for package in configured.split(",") if package.strip()
    )
    if not task_packages:
        raise ValueError("PGPT_ARQ_TASK_PACKAGES must configure at least one package")
    return task_packages


def _queue_name() -> str:
    queue = os.environ.get("PGPT_ARQ_QUEUE", "").strip()
    if not queue:
        raise ValueError("PGPT_ARQ_QUEUE must configure a queue")
    return get_queue_name(queue)


def _keep_result_seconds(current_settings: Settings) -> int:
    default = current_settings.scheduler.chat.callback_timeout_seconds + 300
    return int(os.environ.get("PGPT_ARQ_KEEP_RESULT", str(default)))


def run_arq_worker(
    *,
    settings_resolver: Callable[[], Settings] = settings,
    startup_hook: StartupShutdown = startup,
    shutdown_hook: StartupShutdown = shutdown,
) -> None:
    app_module = os.environ.get("PGPT_WORKER_APP_MODULE", "private_gpt")
    current_settings = settings_resolver()
    task_packages = _task_packages()
    queue_name = _queue_name()
    max_jobs = int(os.environ.get("PGPT_ARQ_MAX_JOBS", str(_default_concurrency())))
    job_timeout = int(os.environ.get("PGPT_ARQ_JOB_TIMEOUT", "21600"))
    keep_result = _keep_result_seconds(current_settings)
    api_enabled = os.environ.get("API_ENABLED", "true").lower() == "true"
    api_port = os.environ.get("API_PORT", "8091")
    healthcheck_app = f"{app_module}.arq.healthcheck:app"
    procs: list[subprocess.Popen[bytes]] = []

    def _cleanup(signum: int = 0, frame: object | None = None) -> None:
        del frame
        for proc in procs:
            proc.terminate()
        for proc in procs:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if signum:
            raise SystemExit(0)

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    if api_enabled:
        print(f"Starting arq worker healthcheck on port {api_port}")
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    healthcheck_app,
                    "--host",
                    "0.0.0.0",
                    "--port",
                    api_port,
                    "--no-access-log",
                    "--log-level",
                    "critical",
                ]
            )
        )

    async def _main() -> None:
        worker = Worker(
            functions=autodiscover_registered_tasks(*task_packages),
            queue_name=queue_name,
            redis_settings=get_redis_settings(current_settings),
            on_startup=startup_hook,
            on_shutdown=shutdown_hook,
            on_job_end=on_job_end,
            handle_signals=False,
            allow_abort_jobs=True,
            max_jobs=max_jobs,
            max_tries=1,
            retry_jobs=False,
            keep_result=keep_result,
            job_timeout=job_timeout,
            health_check_interval=30,
            job_completion_wait=5,
        )
        loop = asyncio.get_running_loop()
        stop_event = asyncio.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop_event.set)
        worker_task = asyncio.create_task(worker.async_run())
        stop_task = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait(
            {worker_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if stop_task in done and not worker_task.done():
            await worker.close()
            with contextlib.suppress(asyncio.CancelledError):
                await worker_task
        for task in pending:
            task.cancel()

    print(
        f"Starting arq worker queue={queue_name} task_packages={','.join(task_packages)} "
        f"max_jobs={max_jobs} job_timeout={job_timeout} keep_result={keep_result}"
    )
    try:
        asyncio.run(_main())
    finally:
        _cleanup()


if __name__ == "__main__":
    run_arq_worker()
