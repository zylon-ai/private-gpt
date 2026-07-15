import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import (
    Any,
)

from retry_async import retry as retry_untyped  # type: ignore
from retry_async.api import retry_call_async  # type: ignore

retry_logger = logging.getLogger(__name__)

AsyncRetryFn = Callable[..., Awaitable[Any]]


def retry(
    exceptions: Any = Exception,
    *,
    is_async: bool = False,
    tries: int = -1,
    delay: float = 0,
    max_delay: float | None = None,
    backoff: float = 1,
    jitter: float | tuple[float, float] = 0,
    logger: logging.Logger = retry_logger,
) -> Callable[..., Any]:
    wrapped = retry_untyped(
        exceptions=exceptions,
        is_async=is_async,
        tries=tries,
        delay=delay,
        max_delay=max_delay,
        backoff=backoff,
        jitter=jitter,
        logger=logger,
    )
    return wrapped  # type: ignore


@asynccontextmanager
async def retry_context(
    exceptions: Any = Exception,
    *,
    tries: int = -1,
    delay: float = 0,
    max_delay: float | None = None,
    backoff: float = 2.0,
    jitter: float | tuple[float, float] = 0,
    logger: logging.Logger = retry_logger,
) -> AsyncGenerator[AsyncRetryFn, None]:
    async def retry_func(
        func: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> Any:
        result: Any = await retry_call_async(
            f=func,
            fargs=args,
            fkwargs=kwargs,
            exceptions=exceptions,
            tries=tries,
            delay=delay,
            max_delay=max_delay,
            backoff=backoff,
            jitter=jitter,
            logger=logger,
        )
        return result

    try:
        yield retry_func
    finally:
        pass
