import asyncio
import functools
import logging
import time
from collections.abc import Callable, Coroutine
from typing import (
    Any,
    ParamSpec,
    TypeVar,
    cast,
)

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def timeit(func: Callable[P, T]) -> Callable[P, T]:
    """Decorator to measure execution time of both sync and async functions."""
    function_name = getattr(func, "__name__", func.__class__.__name__)

    @functools.wraps(func)
    def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed_time = time.perf_counter() - start_time
            logger.info(
                f"{function_name} executed in {elapsed_time:.4f} seconds",
                extra={
                    "function": function_name,
                    "execution_time": elapsed_time,
                    "function_module": func.__module__,  # Changed from "module"
                },
            )

    @functools.wraps(func)
    async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        start_time = time.perf_counter()
        try:
            result = await cast(Coroutine[Any, Any, T], func(*args, **kwargs))
            return result
        finally:
            elapsed_time = time.perf_counter() - start_time
            logger.info(
                f"{function_name} executed in {elapsed_time:.4f} seconds",
                extra={
                    "function": function_name,
                    "execution_time": elapsed_time,
                    "function_module": func.__module__,  # Changed from "module"
                },
            )

    if asyncio.iscoroutinefunction(func):
        return cast(Callable[P, T], async_wrapper)
    return cast(Callable[P, T], sync_wrapper)
