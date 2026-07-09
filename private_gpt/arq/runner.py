import asyncio
import importlib
import os
import signal
import subprocess
import sys

from arq.worker import Worker

from private_gpt.arq.settings import (
    get_health_check_key,
    get_queue_name,
    get_redis_settings,
)
from private_gpt.settings.settings import Settings, settings


def _default_concurrency() -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1:
        return 1
    return 1 << (cpu_count.bit_length() - 1)


def run_arq_worker() -> None:
    app_module = os.environ.get("PGPT_WORKER_APP_MODULE", "private_gpt")
    worker_module = importlib.import_module(f"{app_module}.arq.worker")
    settings_module = importlib.import_module(f"{app_module}.settings.settings")
    settings_resolver = getattr(settings_module, "settings", settings)
    current_settings: Settings = settings_resolver()
    max_jobs = int(os.environ.get("PGPT_ARQ_MAX_JOBS", str(_default_concurrency())))
    job_timeout = int(os.environ.get("PGPT_ARQ_JOB_TIMEOUT", "21600"))
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
            functions=worker_module.functions,
            queue_name=get_queue_name(current_settings),
            redis_settings=get_redis_settings(current_settings),
            on_startup=worker_module.startup,
            on_shutdown=worker_module.shutdown,
            on_job_end=worker_module.on_job_end,
            handle_signals=False,
            allow_abort_jobs=False,
            max_jobs=max_jobs,
            max_tries=1,
            retry_jobs=False,
            keep_result=0,
            job_timeout=job_timeout,
            health_check_key=get_health_check_key(current_settings),
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
            await worker_task
        for task in pending:
            task.cancel()

    print(
        f"Starting arq worker queue={get_queue_name(current_settings)} max_jobs={max_jobs} job_timeout={job_timeout}"
    )
    try:
        asyncio.run(_main())
    finally:
        _cleanup()


if __name__ == "__main__":
    run_arq_worker()
