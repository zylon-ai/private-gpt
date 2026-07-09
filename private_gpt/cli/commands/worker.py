import os
import signal
import subprocess
import sys

import typer

from private_gpt.cli.commands.arq_worker import arq_worker_command


def _build_flower_args(mode: str) -> list[str]:
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
    if mode == "mixed":
        args.append("--logging=none")
    if persistent.lower() == "true":
        args.append("--persistent=True")
        if persistent_db:
            args.append(f"--db={persistent_db}")
        if max_tasks:
            args.append(f"--max-tasks={max_tasks}")
    return args


def _build_worker_args() -> list[str]:
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
        args.append("--max-tasks-per-child=1000000")

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


def worker_command() -> None:
    """Start a background worker process.

    Behaviour is fully controlled through environment variables.
    """
    mode = os.environ.get("PGPT_WORKER_MODE", "mixed").strip().lower()

    if mode == "arq":
        arq_worker_command()
        return

    app_module = os.environ.get("PGPT_WORKER_APP_MODULE", "private_gpt")
    celery_app = f"{app_module}.celery"
    healthcheck_app = f"{app_module}.celery.healthcheck:app"
    procs: list[subprocess.Popen[bytes]] = []

    def _cleanup(signum: int = 0, frame: object = None) -> None:
        typer.echo("Shutting down services...")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    if mode in ("flower", "mixed"):
        flower_args = _build_flower_args(mode)
        typer.echo(
            f"Starting flower on port {os.environ.get('PGPT_FLOWER_PORT', '5555')}"
        )
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "celery",
                    "--app",
                    celery_app,
                    "flower",
                    *flower_args,
                ]
            )
        )

    if mode in ("worker", "mixed"):
        worker_args = _build_worker_args()
        typer.echo(f"Starting celery worker with args: {' '.join(worker_args)}")
        # The PGPT_STATEFUL_WORKER_TYPE env var triggers eager warm-up in the
        # worker process via bootsteps.py.
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "celery",
                    "--app",
                    celery_app,
                    "worker",
                    *worker_args,
                ]
            )
        )

    if os.environ.get("API_ENABLED", "true").lower() == "true":
        api_port = os.environ.get("API_PORT", "8090")
        typer.echo(f"Starting healthcheck server on port {api_port}")
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

    if not procs:
        typer.echo(f"No processes started. Check PGPT_WORKER_MODE={mode!r}", err=True)
        raise SystemExit(1)

    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        _cleanup()
