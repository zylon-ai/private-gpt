from arq.connections import RedisSettings

from private_gpt.settings.settings import Settings

CHAT_HEALTH_CHECK_KEY_PREFIX = "private_gpt:arq:health"
CHAT_TASK_NAME = "private_gpt.chat.run"
CHAT_QUEUE_PREFIX = "private_gpt:arq:queue"


def get_queue_name(settings: Settings) -> str:
    return f"{CHAT_QUEUE_PREFIX}:{settings.scheduler.chat.celery_queue}"


def get_health_check_key(settings: Settings) -> str:
    return f"{CHAT_HEALTH_CHECK_KEY_PREFIX}:{settings.scheduler.chat.celery_queue}"


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
