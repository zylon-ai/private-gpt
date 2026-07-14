from arq.connections import RedisSettings

from private_gpt.settings.settings import Settings

QUEUE_PREFIX = "private_gpt:arq:queue"


def get_queue_name(queue: str) -> str:
    return f"{QUEUE_PREFIX}:{queue}"


def get_redis_settings(settings: Settings) -> RedisSettings:
    database = int(settings.redis.database or 0) + 8
    host = settings.redis.host
    if ":" in host:
        redis_host, redis_port = host.rsplit(":", 1)
    else:
        redis_host, redis_port = host, "6379"

    return RedisSettings(
        host=redis_host,
        port=int(redis_port),
        database=database,
        username=settings.redis.username,
        password=settings.redis.password,
    )
