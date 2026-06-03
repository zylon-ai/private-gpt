import logging
import os
import signal
from pathlib import Path

import typer
import uvicorn

from private_gpt.settings.settings import settings
from private_gpt.settings.settings_loader import active_profiles

logger = logging.getLogger(__name__)

PID_FILE_OPTION = typer.Option(
    None, "--pid-file", help="Write PID to file (for systemd / launchd)"
)


def serve_command(
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int | None = typer.Option(None, help="HTTP port (default: from settings)"),
    reload: bool = typer.Option(
        False, "--reload", help="Enable auto-reload for development"
    ),
    log_level: str = typer.Option(
        "info", "--log-level", help="debug | info | warn | error"
    ),
    pid_file: Path | None = PID_FILE_OPTION,
) -> None:
    """Start the HTTP server."""
    s = settings()
    resolved_port = port if port is not None else s.server.port
    logger.info(
        "Starting server with profiles=%s on port %s", active_profiles, resolved_port
    )

    if pid_file and pid_file.exists():
        try:
            existing_pid = int(pid_file.read_text().strip())
            os.kill(existing_pid, 0)
            typer.echo(f"Server is already running with PID {existing_pid}", err=True)
            raise SystemExit(1)
        except ProcessLookupError:
            pass  # stale PID file

    if pid_file:
        pid_file.parent.mkdir(parents=True, exist_ok=True)
        pid_file.write_text(str(os.getpid()))

    def _on_sigterm(signum: int, frame: object) -> None:
        if pid_file and pid_file.exists():
            pid_file.unlink(missing_ok=True)
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        uvicorn.run(
            "private_gpt.main:app",
            host=host,
            port=resolved_port,
            reload=reload,
            log_level=log_level,
            log_config=None,
            loop="asyncio",
        )
    finally:
        if pid_file and pid_file.exists():
            pid_file.unlink(missing_ok=True)
