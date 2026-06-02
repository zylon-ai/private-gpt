import asyncio
import logging
import random
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Coroutine
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")


async def _clean_up_tasks(tasks: list[asyncio.Task[Any]]) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def bounded_concurrent_execute(
    tasks: list[Coroutine[T, Any, R]],
    concurrency_limit: int | None = None,
    jitter: tuple[float, float] | None = None,
) -> list[R]:
    """Execute coroutines with configurable concurrency limit and jitter.

    Args:
        tasks: List of coroutines to execute
        concurrency_limit: Maximum number of concurrent tasks (None for unlimited)
        jitter: Tuple of (min, max) seconds to randomly delay task execution

    Returns:
        List of results in the same order as input coroutines
    """
    effective_limit = len(tasks) if concurrency_limit is None else concurrency_limit
    semaphore = asyncio.Semaphore(effective_limit)
    scheduled_tasks: list[asyncio.Task[R]] = []

    async def execute_with_control(coro: Coroutine[T, Any, R]) -> R:
        async with semaphore:
            if jitter:
                min_delay, max_delay = jitter
                delay = min_delay + random.random() * (max_delay - min_delay)
                logger.debug(f"Delaying task by {delay:.2f} seconds")
                await asyncio.sleep(delay)
            return await coro

    try:
        scheduled_tasks = [asyncio.create_task(execute_with_control(c)) for c in tasks]
        return await asyncio.gather(*scheduled_tasks)  # type: ignore
    except asyncio.CancelledError:
        await _clean_up_tasks(scheduled_tasks)
        raise
    finally:
        await _clean_up_tasks(scheduled_tasks)


async def map_elements_in_parallel(
    items: AsyncIterable[T],
    processor: Callable[[T], Coroutine[Any, Any, R]],
    *,
    num_workers: int | None = None,
) -> AsyncGenerator[R, None]:
    tasks: set[asyncio.Task[R]] = set()

    try:
        if num_workers is None:
            async for item in items:
                tasks.add(asyncio.create_task(processor(item)))
            for task in asyncio.as_completed(tasks):
                yield await task
            return

        semaphore = asyncio.Semaphore(num_workers)

        async def sem_processor(item: T) -> R:
            async with semaphore:
                return await processor(item)

        async for item in items:
            task = asyncio.create_task(sem_processor(item))
            tasks.add(task)
            if len(tasks) >= num_workers:
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for completed in done:
                    tasks.remove(completed)
                    yield completed.result()

        for task in asyncio.as_completed(tasks):
            yield await task
    except asyncio.CancelledError:
        await _clean_up_tasks(list(tasks))
        raise
    finally:
        await _clean_up_tasks(list(tasks))
