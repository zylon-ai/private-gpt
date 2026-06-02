import asyncio
import contextlib
import threading
from collections.abc import Callable
from typing import Any

import nest_asyncio  # type: ignore
from celery.app.task import Task
from celery.backends.redis import RedisBackend  # type: ignore
from celery.exceptions import Retry, SoftTimeLimitExceeded
from celery.utils.log import get_task_logger

from private_gpt.celery.callback import task_after_return
from private_gpt.celery.config import celery_settings
from private_gpt.di import clean_global_injector, get_global_injector

logger = get_task_logger(__name__)
logger.setLevel("DEBUG")

REDIS_FAILURE_RETRIES_PREFIX = "failure_retries"
REDIS_FAILURE_RETRIES_EXPIRY = 86400


class RedisFailureRetryTracker:
    """Redis implementation for tracking failure retries."""

    _key_prefix: str
    _expiry: int

    def __init__(
        self,
        key_prefix: str = REDIS_FAILURE_RETRIES_PREFIX,
        expiry: int = REDIS_FAILURE_RETRIES_EXPIRY,
    ):
        self._key_prefix = key_prefix
        self._expiry = expiry

    def get(self, task_id: str) -> int:
        from celery import current_app  # type: ignore

        key = self._get_key(task_id)
        backend = current_app.backend
        assert isinstance(backend, RedisBackend)
        return int(backend.client.get(key) or 0)

    def increment(self, task_id: str) -> int:
        from celery import current_app  # type: ignore

        key = self._get_key(task_id)
        backend = current_app.backend
        assert isinstance(backend, RedisBackend)
        pipe = backend.client.pipeline()
        pipe.incr(key)
        pipe.expire(key, self._expiry)
        result = pipe.execute()
        return int(result[0])

    def decrement(self, task_id: str) -> int:
        from celery import current_app  # type: ignore

        key = self._get_key(task_id)
        backend = current_app.backend
        assert isinstance(backend, RedisBackend)
        pipe = backend.client.pipeline()
        pipe.decr(key)
        pipe.expire(key, self._expiry)
        result = pipe.execute()
        return int(result[0])

    def cleanup(self, task_id: str) -> None:
        from celery import current_app  # type: ignore

        key = self._get_key(task_id)
        backend = current_app.backend
        assert isinstance(backend, RedisBackend)
        backend.delete(key)

    def _get_key(self, task_id: str) -> str:
        return f"{self._key_prefix}_{task_id}"


def create_failure_retry_tracker() -> RedisFailureRetryTracker | None:
    """Create a new failure retry tracker."""
    if not celery_settings.acks_late:
        # Failure retries are only needed when acks_late is enabled
        return None

    providers = {
        "redis": RedisFailureRetryTracker,
    }
    provider = providers.get(celery_settings.backend_mode)
    return provider() if provider else None


class MaxFailureRetriesExceeded(Exception):
    """Raised when maximum failure retries are exceeded."""

    pass


class _BackgroundTask(Task):  # type: ignore
    """Task with controlled and failure retry capabilities."""

    abstract = True

    # == General task settings ==
    after_return = task_after_return  # type: ignore
    rollback_fn: Callable[..., Any] | None = None

    # === Controlled retries ===
    max_controlled_retries: int = 10
    default_retry_delay: int = 20
    retry_backoff: bool = True

    # === Failure retries ===
    max_failure_retries: int = 3

    def __init__(
        self,
    ) -> None:
        super().__init__()
        self._retry_tracker = create_failure_retry_tracker()

    def get_task_id(self) -> str | None:
        """Get current task ID if available."""
        return self.request.id if hasattr(self, "request") else None

    def is_controlled_retry(self) -> bool:
        """Check if current execution is a controlled retry."""
        return bool(getattr(self.request, "is_retry", False))

    def get_controlled_retry_count(self) -> int:
        """Get number of controlled retries."""
        return getattr(self.request, "retries", 0)

    def get_failure_retry_count(self) -> int:
        """Get number of failure retries."""
        task_id = self.get_task_id()
        return (
            self._retry_tracker.get(task_id) if self._retry_tracker and task_id else 1
        )

    def handle_max_failure_retries_exceeded(self, retry_count: int) -> None:
        """Handle when maximum failure retries are exceeded."""
        if not self._retry_tracker:
            return

        task_id = self.get_task_id()
        if task_id:
            self._retry_tracker.cleanup(task_id)
            logger.debug(f"Cleaned up retry counter after max retries for {self.name}")

        raise MaxFailureRetriesExceeded(
            f"Maximum failure retries ({retry_count - 1}/{self.max_failure_retries}) "
            f"exceeded for {self.name}"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute task with retry tracking."""
        task_id = self.get_task_id()
        if not task_id:
            return super().__call__(*args, **kwargs)

        if self.is_controlled_retry():
            logger.debug(
                f"Executing controlled retry for {self.name} "
                f"(Retry {self.get_controlled_retry_count()}/{self.max_controlled_retries})"
            )
        elif self._retry_tracker:
            failure_retries = self._retry_tracker.increment(task_id)
            if failure_retries > self.max_failure_retries:
                logger.error(
                    f"Task {self.name} exceeded maximum failure retries "
                    f"({self.max_failure_retries})"
                )
                return self.handle_max_failure_retries_exceeded(failure_retries)
            logger.debug(
                f"Executing {self.name} "
                f"(Failure retries: {failure_retries - 1}/{self.max_failure_retries}, "
                f"Controlled retries: {self.get_controlled_retry_count()}/{self.max_controlled_retries})"
            )

        try:
            result = super().__call__(*args, **kwargs)
            if self._retry_tracker:
                self._retry_tracker.cleanup(task_id)
            return result
        except Exception as e:
            if self._retry_tracker and not isinstance(e, MaxFailureRetriesExceeded):
                self._retry_tracker.cleanup(task_id)
            if isinstance(e, SoftTimeLimitExceeded):
                # Rollback on soft time limit exceeded
                logger.error(f"Soft time limit exceeded for {self.name}")
            if self.rollback_fn:
                self.rollback_fn(*args, **kwargs)
            raise

    def on_retry(
        self, exc: Exception, task_id: str, args: Any, kwargs: Any, einfo: Any
    ) -> None:
        """Handle controlled retry event."""
        if self._retry_tracker and task_id:
            self._retry_tracker.decrement(task_id)
            logger.debug(
                f"Decremented failure retry counter for controlled retry of {self.name}"
            )
        super().on_retry(exc, task_id, args, kwargs, einfo)

    def on_failure(
        self, exc: Exception, task_id: str, args: Any, kwargs: Any, einfo: Any
    ) -> None:
        """Handle task failure event."""
        if self._retry_tracker and not isinstance(
            exc, Retry | MaxFailureRetriesExceeded
        ):
            retry_count = self.get_failure_retry_count()
            logger.error(
                f"Task {self.name} failed on failure retry "
                f"{retry_count}/{self.max_failure_retries}. Error: {exc!s}"
            )
        super().on_failure(exc, task_id, args, kwargs, einfo)


class AsyncBackgroundTask(_BackgroundTask):
    abstract = True

    async def _run(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the task logic, handling both sync and async implementations."""
        run_method = self.run

        if run_method is None:
            raise NotImplementedError("Subclass must implement 'run' method")

        # Inject injector
        get_global_injector(allow_to_generate_new_injectors=True)

        if asyncio.iscoroutinefunction(run_method):
            result = await run_method(*args, **kwargs)
        else:
            result = run_method(*args, **kwargs)

        if asyncio.iscoroutine(result):
            result = await result

        return result

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        loop: asyncio.AbstractEventLoop | None = None
        thr: threading.Thread | None = None
        try:
            loop = asyncio.new_event_loop()
            nest_asyncio.apply()

            thr = threading.Thread(target=loop.run_forever, daemon=True)
            if not thr.is_alive():
                thr.start()
            coro = self._run(*args, **kwargs)
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()
        except Exception as e:
            if self.rollback_fn:
                self.rollback_fn(*args, **kwargs)
            raise e
        finally:
            if loop:
                clean_global_injector(loop)
                with contextlib.suppress(RuntimeError):
                    # Stop the loop if it's still running
                    loop.call_soon_threadsafe(loop.stop)
            if thr:
                thr.join(timeout=5.0)

            if loop:
                clean_global_injector(loop)
