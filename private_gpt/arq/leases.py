from __future__ import annotations

import os

from arq.constants import in_progress_key_prefix

WORKER_LEASE_KEY_PREFIX = "private_gpt:arq:worker-lease:"
IN_PROGRESS_OWNER_PREFIX = "private_gpt:arq:worker:"
DEFAULT_WORKER_LEASE_SECONDS = 15


def env_seconds(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def worker_lease_seconds() -> int:
    return env_seconds(
        "PGPT_ARQ_WORKER_LEASE_SECONDS",
        DEFAULT_WORKER_LEASE_SECONDS,
    )


def worker_lease_key(worker_id: str) -> str:
    return f"{WORKER_LEASE_KEY_PREFIX}{worker_id}"


def in_progress_key(job_id: str) -> str:
    return f"{in_progress_key_prefix}{job_id}"


def encode_in_progress_owner(worker_id: str) -> bytes:
    return f"{IN_PROGRESS_OWNER_PREFIX}{worker_id}".encode()


def decode_in_progress_owner(value: bytes | str | None) -> str | None:
    if value is None:
        return None
    decoded = value.decode() if isinstance(value, bytes) else value
    if not decoded.startswith(IN_PROGRESS_OWNER_PREFIX):
        return None
    worker_id = decoded.removeprefix(IN_PROGRESS_OWNER_PREFIX)
    return worker_id or None
