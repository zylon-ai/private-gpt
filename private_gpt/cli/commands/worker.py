from private_gpt.worker import run_private_gpt_worker


def worker_command() -> None:
    """Start the worker mode selected by ``PGPT_WORKER_MODE``."""
    run_private_gpt_worker()
