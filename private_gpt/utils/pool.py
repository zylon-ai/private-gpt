import asyncio
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Generic, TypeVar

_T = TypeVar("_T", bound=Any)


class SimplePool(ABC, Generic[_T]):
    """A simple pool for managing reusable objects.

    Implementation is thread-safe.
    """

    _pooled_objects: set[_T]
    _active_objects: set[_T]

    _lock: threading.Lock
    _semaphore: threading.Semaphore | None

    def __init__(self, max_size: int | None = None) -> None:
        self._pooled_objects: set[_T] = set()
        self._active_objects: set[_T] = set()
        self._lock = threading.Lock()
        self._semaphore = (
            threading.Semaphore(max_size) if max_size is not None else None
        )

    @abstractmethod
    def create(self) -> _T:
        """Create a new object to be pooled."""
        pass

    def destroy(self, obj: _T) -> None:
        """Destroy an object. Override for custom cleanup."""
        pass

    def acquire(self) -> _T:
        """Acquire an object from the pool, creating a new one if necessary."""
        if self._semaphore:
            self._semaphore.acquire()

        with self._lock:
            if self._pooled_objects:
                obj = self._pooled_objects.pop()
                self._active_objects.add(obj)
                return obj

            try:
                new_object = self.create()
                self._active_objects.add(new_object)
                return new_object
            except Exception:
                if self._semaphore:
                    self._semaphore.release()
                raise

    def validate(self, obj: _T) -> bool:
        """Check if an object is healthy. Override for custom health checks."""
        return obj is not None

    def release(self, obj: _T) -> None:
        """Release an object back to the pool."""
        if not self.validate(obj):
            self.discard(obj)
            return

        with self._lock:
            if obj in self._active_objects:
                self._active_objects.remove(obj)
                self._pooled_objects.add(obj)

        if self._semaphore:
            self._semaphore.release()

    def discard(self, obj: _T) -> None:
        """Discard an object without returning it to the pool.

        After this method is called, we need to be sure that the semaphore is released
        """
        with self._lock:
            if obj in self._active_objects:
                self._active_objects.remove(obj)
            if obj in self._pooled_objects:
                self._pooled_objects.remove(obj)
            self.destroy(obj)

    def close(self) -> None:
        """Close pool and clean up all objects."""
        with self._lock:
            for obj in self._pooled_objects | self._active_objects:
                self.destroy(obj)
            self._pooled_objects.clear()
            self._active_objects.clear()

    @contextmanager
    def acquire_context(self) -> Generator[_T, None, None]:
        """Context manager for acquiring an object from the pool."""
        obj = self.acquire()
        try:
            yield obj
        except Exception as e:
            self.discard(obj)
            raise e
        finally:
            self.release(obj)


class AsyncSimplePool(ABC, Generic[_T]):
    """A simple async pool for managing reusable objects.

    Implementation is async-safe.
    """

    _pooled_objects: set[_T]
    _active_objects: set[_T]

    _lock: asyncio.Lock
    _semaphore: asyncio.Semaphore | None

    def __init__(self, max_size: int | None = None) -> None:
        self._pooled_objects: set[_T] = set()
        self._active_objects: set[_T] = set()
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_size) if max_size is not None else None

    @abstractmethod
    async def create(self) -> _T:
        """Create a new object to be pooled."""
        pass

    async def destroy(self, obj: _T) -> None:
        """Destroy an object. Override for custom cleanup."""
        pass

    async def acquire(self) -> _T:
        """Acquire an object from the pool, creating a new one if necessary."""
        if self._semaphore:
            await self._semaphore.acquire()

        async with self._lock:
            if self._pooled_objects:
                obj = self._pooled_objects.pop()
                self._active_objects.add(obj)
                return obj

            try:
                new_object = await self.create()
                self._active_objects.add(new_object)
                return new_object
            except Exception:
                if self._semaphore:
                    self._semaphore.release()
                raise

    async def validate(self, obj: _T) -> bool:
        """Check if an object is healthy. Override for custom health checks."""
        return obj is not None

    async def release(self, obj: _T) -> None:
        """Release an object back to the pool."""
        if not await self.validate(obj):
            await self.discard(obj)
            return

        async with self._lock:
            if obj in self._active_objects:
                self._active_objects.remove(obj)
                self._pooled_objects.add(obj)

        if self._semaphore:
            self._semaphore.release()

    async def discard(self, obj: _T) -> None:
        """Discard an object without returning it to the pool.

        After this method is called, we need to be sure that the semaphore is released
        """
        async with self._lock:
            if obj in self._active_objects:
                self._active_objects.remove(obj)
            if obj in self._pooled_objects:
                self._pooled_objects.remove(obj)
            await self.destroy(obj)

    async def close(self) -> None:
        """Close pool and clean up all objects."""
        async with self._lock:
            for obj in self._pooled_objects | self._active_objects:
                await self.destroy(obj)
            self._pooled_objects.clear()
            self._active_objects.clear()

    @asynccontextmanager
    async def acquire_context(self) -> AsyncGenerator[_T]:
        """Context manager for acquiring an object from the pool."""
        obj = await self.acquire()
        try:
            yield obj
        except Exception as e:
            await self.discard(obj)
            raise e
        finally:
            await self.release(obj)
