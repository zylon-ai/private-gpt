from collections.abc import Sequence

from private_gpt.worker.modes import register_private_gpt_worker_modes
from private_gpt.worker.registry import register_worker_mode, run_worker


def run_private_gpt_worker(args: Sequence[str] = ()) -> None:
    register_private_gpt_worker_modes()
    run_worker(args)


__all__ = ["register_worker_mode", "run_private_gpt_worker", "run_worker"]
