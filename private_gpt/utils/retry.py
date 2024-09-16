import logging
from collections.abc import Callable
from typing import Any

from retry_async import retry as retry_untyped  # type: ignore

retry_logger = logging.getLogger(__name__)


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
