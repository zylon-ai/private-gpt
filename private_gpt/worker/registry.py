import os
from collections.abc import Callable, Sequence

WorkerModeHandler = Callable[[Sequence[str]], None]

_worker_modes: dict[str, WorkerModeHandler] = {}


def register_worker_mode(name: str, handler: WorkerModeHandler) -> None:
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Worker mode name cannot be empty")
    _worker_modes[normalized_name] = handler


def get_worker_mode(name: str) -> WorkerModeHandler:
    normalized_name = name.strip().lower()
    try:
        return _worker_modes[normalized_name]
    except KeyError as exc:
        supported_modes = ", ".join(sorted(_worker_modes))
        raise ValueError(
            f"Unsupported PGPT_WORKER_MODE={name!r}. Registered modes: {supported_modes}"
        ) from exc


def run_worker(args: Sequence[str] = ()) -> None:
    mode = os.environ.get("PGPT_WORKER_MODE", "").strip()
    if not mode:
        raise ValueError("PGPT_WORKER_MODE is required")
    get_worker_mode(mode)(args)
