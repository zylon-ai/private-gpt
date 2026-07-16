import asyncio
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Generic, TypeVar

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
