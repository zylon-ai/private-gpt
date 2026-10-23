import hashlib
import logging
import pickle
import threading
import time
from collections import OrderedDict
from typing import Any, Protocol, cast

from injector import inject, singleton

from private_gpt.settings.settings import Settings
from private_gpt.utils.dependencies import format_missing_dependency_message

logger = logging.getLogger(__name__)


class Cache(Protocol):
    def get(self, namespace: str, key: str) -> Any | None: ...

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None: ...

    def delete(self, namespace: str, key: str) -> None: ...


class MemoryCache:
    def __init__(self, max_entries: int = 1000) -> None:
        self._max_entries = max_entries
        self._values: OrderedDict[str, tuple[float | None, Any]] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _key(namespace: str, key: str) -> str:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return f"{namespace}:{digest}"

    def get(self, namespace: str, key: str) -> Any | None:
        cache_key = self._key(namespace, key)
        with self._lock:
            entry = self._values.get(cache_key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at is not None and expires_at <= time.monotonic():
                del self._values[cache_key]
                return None
            self._values.move_to_end(cache_key)
            return value

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        cache_key = self._key(namespace, key)
        expires_at = time.monotonic() + ttl_seconds if ttl_seconds is not None else None
        with self._lock:
            self._values[cache_key] = (expires_at, value)
            self._values.move_to_end(cache_key)
            while len(self._values) > self._max_entries:
                self._values.popitem(last=False)

    def delete(self, namespace: str, key: str) -> None:
        with self._lock:
            self._values.pop(self._key(namespace, key), None)


class RedisCache:
    def __init__(self, settings: Settings) -> None:
        try:
            import redis
        except ImportError as error:
            raise ImportError(
                format_missing_dependency_message("Redis cache", extras="redis")
            ) from error

        redis_database = (
            settings.cache.redis_database
            if settings.cache.redis_database is not None
            else int(settings.redis.database or 0)
        )
        self._prefix = settings.cache.key_prefix
        self._client = redis.Redis.from_url(
            settings.redis.url,
            db=redis_database,
            decode_responses=False,
        )

    def _key(self, namespace: str, key: str) -> str:
        digest = hashlib.sha256(key.encode()).hexdigest()
        return f"{self._prefix}:{namespace}:{digest}"

    def get(self, namespace: str, key: str) -> Any | None:
        value = self._client.get(self._key(namespace, key))
        return pickle.loads(cast(bytes, value)) if value is not None else None

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        self._client.set(
            self._key(namespace, key),
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
            ex=ttl_seconds,
        )

    def delete(self, namespace: str, key: str) -> None:
        self._client.delete(self._key(namespace, key))

    def close(self) -> None:
        self._client.close()  # type: ignore[no-untyped-call]


@singleton
class CacheService:
    @inject
    def __init__(self, settings: Settings) -> None:
        self._default_ttl = settings.cache.ttl_seconds
        self._memory = MemoryCache(settings.cache.max_entries)
        self._distributed = (
            RedisCache(settings) if settings.cache.provider == "redis" else None
        )

    def get(self, namespace: str, key: str) -> Any | None:
        value = self._memory.get(namespace, key)
        if value is not None or self._distributed is None:
            return value
        try:
            value = self._distributed.get(namespace, key)
        except Exception:
            logger.warning("Distributed cache read failed", exc_info=True)
            return None
        if value is not None:
            self._memory.set(namespace, key, value, self._default_ttl)
        return value

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        ttl_seconds: int | None = None,
    ) -> None:
        effective_ttl = self._default_ttl if ttl_seconds is None else ttl_seconds
        self._memory.set(namespace, key, value, effective_ttl)
        if self._distributed is not None:
            try:
                self._distributed.set(namespace, key, value, effective_ttl)
            except Exception:
                logger.warning("Distributed cache write failed", exc_info=True)

    def delete(self, namespace: str, key: str) -> None:
        self._memory.delete(namespace, key)
        if self._distributed is not None:
            try:
                self._distributed.delete(namespace, key)
            except Exception:
                logger.warning("Distributed cache delete failed", exc_info=True)

    def close(self) -> None:
        if self._distributed is not None:
            self._distributed.close()
