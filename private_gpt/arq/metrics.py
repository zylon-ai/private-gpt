"""Prometheus scrape endpoint for the ARQ worker process.

``/health`` runs in a child uvicorn and cannot see this event loop. This server
runs in a daemon thread inside the worker, so ``/metrics`` and ``/debug`` still
answer when the loop is blocked.

Not an OTLP push. Grafana Alloy / OpenTelemetry collectors scrape Prometheus
text the same way they scrape Triton.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 9464
_SERVER: ThreadingHTTPServer | None = None


def _env_port() -> int | None:
    raw = os.environ.get("PGPT_ARQ_METRICS_PORT", str(_DEFAULT_PORT)).strip().lower()
    if raw in {"0", "off", "false", "no"}:
        return None
    if raw == "":
        return _DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError:
        return _DEFAULT_PORT
    if port <= 0:
        return None
    return port


def render_prometheus(stats: dict[str, Any]) -> str:
    heartbeat = float(stats.get("event_loop_heartbeat_unix") or 0.0)
    heartbeat_age = (time.time() - heartbeat) if heartbeat else 0.0
    lines = [
        "# HELP arq_worker_up 1 while the worker metrics thread is serving",
        "# TYPE arq_worker_up gauge",
        "arq_worker_up 1",
        "# HELP arq_worker_jobs_ongoing In-flight ARQ jobs on this process",
        "# TYPE arq_worker_jobs_ongoing gauge",
        f"arq_worker_jobs_ongoing {int(stats.get('jobs_ongoing') or 0)}",
        "# HELP arq_worker_jobs_max ARQ max_jobs for this process",
        "# TYPE arq_worker_jobs_max gauge",
        f"arq_worker_jobs_max {int(stats.get('jobs_max') or 0)}",
        "# HELP arq_worker_jobs_complete Jobs completed by this process",
        "# TYPE arq_worker_jobs_complete gauge",
        f"arq_worker_jobs_complete {int(stats.get('jobs_complete') or 0)}",
        "# HELP arq_worker_jobs_failed Jobs failed by this process",
        "# TYPE arq_worker_jobs_failed gauge",
        f"arq_worker_jobs_failed {int(stats.get('jobs_failed') or 0)}",
        "# HELP arq_worker_oldest_job_age_seconds Age of the oldest in-flight job",
        "# TYPE arq_worker_oldest_job_age_seconds gauge",
        f"arq_worker_oldest_job_age_seconds {float(stats.get('oldest_job_age_seconds') or 0.0):.3f}",
        "# HELP arq_worker_event_loop_lag_seconds Delay of the 1s loop heartbeat",
        "# TYPE arq_worker_event_loop_lag_seconds gauge",
        f"arq_worker_event_loop_lag_seconds {float(stats.get('event_loop_lag_seconds') or 0.0):.6f}",
        "# HELP arq_worker_event_loop_heartbeat_timestamp_seconds Unix time of last loop heartbeat",
        "# TYPE arq_worker_event_loop_heartbeat_timestamp_seconds gauge",
        f"arq_worker_event_loop_heartbeat_timestamp_seconds {heartbeat:.3f}",
        "# HELP arq_worker_event_loop_heartbeat_age_seconds Seconds since last loop heartbeat",
        "# TYPE arq_worker_event_loop_heartbeat_age_seconds gauge",
        f"arq_worker_event_loop_heartbeat_age_seconds {heartbeat_age:.3f}",
        "# HELP arq_worker_asyncio_tasks Asyncio tasks at last loop heartbeat",
        "# TYPE arq_worker_asyncio_tasks gauge",
        f"arq_worker_asyncio_tasks {int(stats.get('asyncio_tasks') or 0)}",
        "# HELP arq_worker_job_timeout_seconds Configured ARQ job timeout",
        "# TYPE arq_worker_job_timeout_seconds gauge",
        f"arq_worker_job_timeout_seconds {float(stats.get('job_timeout_s') or 0.0):.0f}",
    ]
    return "\n".join(lines) + "\n"


def _debug_payload() -> dict[str, Any]:
    from private_gpt.arq.debug import build_debug_payload

    return build_debug_payload()


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/metrics":
            from private_gpt.arq.debug import collect_worker_stats

            body = render_prometheus(collect_worker_stats()).encode("utf-8")
            self._write(200, "text/plain; version=0.0.4; charset=utf-8", body)
            return
        if path in {"/debug", "/debug/"}:
            body = json.dumps(_debug_payload(), default=str).encode("utf-8")
            self._write(200, "application/json; charset=utf-8", body)
            return
        self._write(404, "text/plain; charset=utf-8", b"not found\n")

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _write(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_metrics_server(port: int | None = None) -> int | None:
    """Serve /metrics and /debug in a daemon thread. port=0 binds ephemeral."""
    global _SERVER
    if _SERVER is not None:
        return int(_SERVER.server_address[1])
    if port is None:
        port = _env_port()
        if port is None:
            return None
    server = ThreadingHTTPServer(("0.0.0.0", port), _Handler)
    _SERVER = server
    thread = threading.Thread(
        target=server.serve_forever,
        name="arq-metrics",
        daemon=True,
    )
    thread.start()
    bound = int(server.server_address[1])
    logger.info("ARQ metrics endpoint http://0.0.0.0:%s/metrics", bound)
    return bound


def reset_metrics_server() -> None:
    """Test helper."""
    global _SERVER
    if _SERVER is None:
        return
    with contextlib.suppress(Exception):
        _SERVER.shutdown()
    with contextlib.suppress(Exception):
        _SERVER.server_close()
    _SERVER = None
