import asyncio

from private_gpt.utils.async_utils import run_sync


async def _answer() -> int:
    await asyncio.sleep(0)
    return 42


def test_run_sync_without_a_running_loop() -> None:
    assert run_sync(_answer()) == 42


async def test_run_sync_inside_a_running_loop() -> None:
    """Sync code reached from async code (e.g. startup) can still run coroutines."""
    assert run_sync(_answer()) == 42


async def test_run_sync_keeps_the_callers_loop_injector() -> None:
    """Workers keep their injector on the loop; run_sync must not lose it."""
    from injector import Injector

    from private_gpt.di import get_global_injector, set_global_injector

    task_injector = Injector()
    set_global_injector(task_injector)

    async def current_injector() -> Injector:
        return get_global_injector(allow_to_generate_new_injectors=False)

    assert run_sync(current_injector()) is task_injector
    assert run_sync(current_injector()) is task_injector
