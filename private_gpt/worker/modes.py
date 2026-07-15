import importlib
import os
import signal
import subprocess
import sys
from collections.abc import Callable, Sequence
from functools import partial

import typer

from private_gpt.settings.settings import CelerySettings, settings
from private_gpt.worker.registry import register_worker_mode


def _app_module() -> str:
    return os.environ.get("PGPT_WORKER_APP_MODULE", "private_gpt")


def _build_flower_args() -> list[str]:
    args: list[str] = []
    url_prefix = os.environ.get("PGPT_FLOWER_URL_PREFIX", "")
    port = os.environ.get("PGPT_FLOWER_PORT", "5555")
    password = os.environ.get("PGPT_FLOWER_PASSWORD", "flower")
    persistent = os.environ.get("PGPT_FLOWER_PERSISTENT", "true")
    persistent_db = os.environ.get("PGPT_FLOWER_PERSISTENT_DB", "celery_flower.db")
    max_tasks = os.environ.get("PGPT_FLOWER_MAXIMUM_TASKS", "1000")

    if url_prefix:
        args.append(f"--url_prefix={url_prefix}")
    if port:
        args.append(f"--port={port}")
    if password:
        args.append(f"--basic_auth=flower:{password}")
    if persistent.lower() == "true":
        args.append("--persistent=True")
        if persistent_db:
            args.append(f"--db={persistent_db}")
        if max_tasks:
            args.append(f"--max-tasks={max_tasks}")
    return args


def _build_celery_args(celery_settings: CelerySettings) -> list[str]:
    args: list[str] = []
    log_level = os.environ.get("PGPT_CELERY_LOG_LEVEL", "info")
    queues = os.environ.get("PGPT_CELERY_QUEUES", "")
    hostname = os.environ.get("PGPT_CELERY_HOSTNAME", "default")
    pool = os.environ.get("PGPT_CELERY_POOL", "prefork")
    concurrency = os.environ.get("PGPT_CELERY_CONCURRENCY", "")
    time_limit = os.environ.get("PGPT_CELERY_TIME_LIMIT", "")
    stateful_type = os.environ.get("PGPT_STATEFUL_WORKER_TYPE", "").strip()

    if log_level:
        args.append(f"--loglevel={log_level}")
    if stateful_type:
        queues = queues or stateful_type
        hostname = hostname or stateful_type
        args.append(f"--max-tasks-per-child={celery_settings.max_tasks_per_child}")
        if celery_settings.max_memory_per_child:
            args.append(
                f"--max-memory-per-child={celery_settings.max_memory_per_child}"
            )
    if queues:
        args.append(f"--queues={queues}")
    if hostname:
        args.append(f"--hostname={hostname}%h")
    if pool:
        args.append(f"--pool={pool}")
    if concurrency:
        args.append(f"--concurrency={concurrency}")
    if time_limit:
        args.append(f"--time-limit={time_limit}")
    return args


def _run_processes(commands: Sequence[Sequence[str]]) -> None:
    processes = [subprocess.Popen(command) for command in commands]

    def cleanup(signum: int = 0, frame: object = None) -> None:
        del signum, frame
        typer.echo("Shutting down services...")
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    try:
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        cleanup()


def _healthcheck_command() -> list[str]:
    return [
        sys.executable,
        "-m",
        "uvicorn",
        f"{_app_module()}.celery.healthcheck:app",
        "--host",
        "0.0.0.0",
        "--port",
        os.environ.get("API_PORT", "8090"),
        "--no-access-log",
        "--log-level",
        "critical",
    ]


def _with_healthcheck(command: list[str]) -> list[list[str]]:
    commands = [command]
    if os.environ.get("API_ENABLED", "true").lower() == "true":
        commands.append(_healthcheck_command())
    return commands


def run_arq(args: Sequence[str]) -> None:
    if args:
        raise ValueError("The arq worker mode does not support arguments yet")
    arq_app = importlib.import_module(f"{_app_module()}.arq")
    arq_app.run_arq_worker()


def run_celery(
    args: Sequence[str],
    *,
    celery_settings_provider: Callable[[], CelerySettings] = lambda: settings().celery,
) -> None:
    celery_args = _build_celery_args(celery_settings_provider())
    typer.echo(f"Starting celery worker with args: {' '.join(celery_args)}")
    command = [
        sys.executable,
        "-m",
        "celery",
        "--app",
        f"{_app_module()}.celery",
        "worker",
        *celery_args,
        *args,
    ]
    _run_processes(_with_healthcheck(command))


def run_flower(args: Sequence[str]) -> None:
    flower_args = _build_flower_args()
    typer.echo(f"Starting flower on port {os.environ.get('PGPT_FLOWER_PORT', '5555')}")
    command = [
        sys.executable,
        "-m",
        "celery",
        "--app",
        f"{_app_module()}.celery",
        "flower",
        *flower_args,
        *args,
    ]
    _run_processes(_with_healthcheck(command))


def register_private_gpt_worker_modes(
    celery_settings_provider: Callable[[], CelerySettings] = lambda: settings().celery,
) -> None:
    register_worker_mode("arq", run_arq)
    register_worker_mode(
        "celery",
        partial(
            run_celery,
            celery_settings_provider=celery_settings_provider,
        ),
    )
    register_worker_mode("flower", run_flower)
