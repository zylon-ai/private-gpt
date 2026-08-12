from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from celery.exceptions import TimeoutError as CeleryTimeoutError

if TYPE_CHECKING:
    from celery.result import AsyncResult


def wait_for_celery_result(
    result: AsyncResult[Any],
    timeout: float | None = None,
    poll_interval: float = 0.1,
) -> Any:
    deadline = time.monotonic() + timeout if timeout is not None else None

    while not result.ready():
        if deadline is None:
            time.sleep(poll_interval)
            continue

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CeleryTimeoutError(f"Task {result.id} timed out")
        time.sleep(min(poll_interval, remaining))

    if result.failed():
        raise result.result
    return result.result
