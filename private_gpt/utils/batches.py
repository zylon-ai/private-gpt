from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Generator,
    Iterable,
)
from typing import (
    Any,
    TypeVar,
)

T = TypeVar("T")


def iter_batch(
    iterable: Iterable[T] | Generator[T],
    size: int,
    stop_condition: Callable[[Any], bool] | None = None,
) -> Generator[list[T], None, None]:
    """Iterate over an iterable in batches with optional stop condition.

    Args:
        iterable: Source iterable to batch
        size: Maximum batch size
        stop_condition: Optional predicate function. If provided and returns True
                       for any item, the current batch is yielded immediately.

    Yields:
        Lists containing batched items
    """
    source_iter = iter(iterable)

    while True:
        batch = []

        try:
            for _ in range(size):
                item = next(source_iter)
                batch.append(item)

                if stop_condition and stop_condition(item):
                    yield batch
                    batch = []
                    break
            else:
                # Loop completed without break (batch is full)
                if batch:
                    yield batch
        except StopIteration:
            # No more items in iterator
            if batch:
                yield batch
            break


async def aiter_batch(
    iterable: AsyncIterable[T] | AsyncGenerator[T, None],
    size: int,
    stop_condition: Callable[[list[T]], bool]
    | Callable[[list[T]], Awaitable[bool]]
    | None = None,
) -> AsyncGenerator[list[T], None]:
    """Async iterate over an async iterable in batches with optional stop condition.

    Args:
        iterable: Source async iterable to batch
        size: Maximum batch size
        stop_condition: Optional predicate function (sync or async).
            If provided and returns True for any item,
            the current batch is yielded immediately.

    Yields:
        Lists containing batched items
    """
    source_iter = aiter(iterable)

    while True:
        batch: list[T] = []

        try:
            for _ in range(size):
                item = await anext(source_iter)
                batch.append(item)

                if stop_condition:
                    should_stop = stop_condition(batch)
                    if isinstance(should_stop, Awaitable):
                        should_stop = await should_stop
                    if should_stop:
                        yield batch
                        batch = []
                        break
            else:
                if batch:
                    yield batch
        except StopAsyncIteration:
            if batch:
                yield batch
            break
