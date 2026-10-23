from collections.abc import Callable

from private_gpt.components.streaming.providers.stream_service import StreamService
from private_gpt.settings.settings import Settings
from private_gpt.utils.dependencies import format_missing_dependency_message

StreamProvider = Callable[[Settings], StreamService]


def _redis_stream_provider(settings: Settings) -> StreamService:
    try:
        from private_gpt.components.streaming.providers.redis_stream_service import (
            LazyRedisClientFactory,
            RedisStreamConfig,
            RedisStreamService,
        )
    except ImportError as e:
        raise ImportError(
            format_missing_dependency_message(
                "Redis streaming",
                extras="redis",
            )
        ) from e

    config = RedisStreamConfig(
        redis_url=settings.redis.url + "/8",
        stream_prefix=settings.stream.stream_prefix,
        status_prefix=settings.stream.status_prefix,
        expiry_seconds=settings.stream.stream_expiration,
        max_stream_length=settings.stream.maximum_stream_length,
        minimum_connections=settings.stream.minimum_connections,
    )
    client = LazyRedisClientFactory.get_instance(config)
    return RedisStreamService(config=config, redis_client=client)


def _memory_stream_provider(settings: Settings) -> StreamService:
    del settings
    from private_gpt.components.streaming.providers.in_memory_stream_service import (
        InMemoryStreamService,
    )

    return InMemoryStreamService()


_PROVIDERS: dict[str, StreamProvider] = {
    "redis": _redis_stream_provider,
    "memory": _memory_stream_provider,
}


def register_stream(name: str, provider: StreamProvider) -> None:
    _PROVIDERS[name] = provider
