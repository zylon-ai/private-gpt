import asyncio
import contextvars
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")


class AsyncIteratorError(Exception):
    """Custom exception for async iterator conversion errors."""

    pass


@dataclass
class OrderedResult(Generic[T]):
    """Helper class to maintain order of results."""

    index: int
    result: T


async def to_async_iterator(
    iterator: Iterator[T],
    transform_fn: Callable[[T], U] | None = None,
    executor: ThreadPoolExecutor | None = None,
    chunk_size: int = 1,
) -> AsyncIterator[U | None]:
    """Convert a synchronous iterator to an async iterator with optional transformation.

    Args:
        iterator: The synchronous iterator to convert
        transform_fn: Optional function to transform each item
        executor: Optional thread pool executor for running iterator operations
        chunk_size: Number of items to process in each async batch

    Yields:
        Transformed items from the iterator

    Raises:
        AsyncIteratorError: If iterator operations fail
    """
    loop = asyncio.get_running_loop()
    internal_executor = executor or ThreadPoolExecutor()

    try:
        while True:
            # Process items in chunks for better performance
            chunk = []
            for _ in range(chunk_size):
                try:

                    def safe_next(it: Iterator[T]) -> T:
                        try:
                            return next(it)
                        except StopIteration:
                            return None  # type: ignore

                    item = await loop.run_in_executor(
                        internal_executor, safe_next, iterator
                    )
                    if item is None:  # Handle StopIteration gracefully
                        break
                    chunk.append(item)
                except Exception as e:
                    raise AsyncIteratorError(
                        f"Iterator next() operation failed: {e!s}"
                    ) from e

            if not chunk:
                break

            # Process the chunk
            for item in chunk:
                try:
                    if transform_fn:
                        # Run transform in executor if it's CPU-intensive
                        result = await loop.run_in_executor(
                            internal_executor, transform_fn, item
                        )
                        yield result
                    else:
                        yield item
                except Exception as e:
                    raise AsyncIteratorError(
                        f"Item transformation failed: {e!s}"
                    ) from e

    except asyncio.CancelledError:
        raise
    except Exception as e:
        raise AsyncIteratorError(f"Async iterator conversion failed: {e!s}") from e
    finally:
        if not executor:
            internal_executor.shutdown(wait=False)


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion from synchronous code.

    ``asyncio.run`` cannot run inside a running event loop. When sync code is
    reached from one (startup hooks, sync library callbacks), the coroutine
    runs on a fresh loop in a worker thread and this call blocks until it is
    done, without patching the caller's loop.
    """
    try:
        caller_loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    context = contextvars.copy_context()
    # Workers keep their injector on the running loop; carry it over so the
    # coroutine resolves the caller's components instead of building new ones.
    from private_gpt.di import _INJECTOR_KEY

    injector = getattr(caller_loop, _INJECTOR_KEY, None)

    async def with_caller_injector() -> T:
        if injector is not None:
            setattr(asyncio.get_running_loop(), _INJECTOR_KEY, injector)
        return await coro

    def run_in_loop() -> T:
        return asyncio.run(with_caller_injector())

    def run() -> T:
        return context.run(run_in_loop)

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="run-sync") as pool:
        return pool.submit(run).result()
