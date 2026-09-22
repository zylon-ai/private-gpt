from unittest.mock import AsyncMock, Mock

import pytest

from private_gpt.components.streaming.providers.models import (
    StreamMetadata,
    StreamStatus,
)
from private_gpt.components.streaming.providers.redis_stream_service import (
    RedisStreamConfig,
    RedisStreamService,
)
from private_gpt.components.streaming.providers.stream_service import Event


@pytest.fixture
def redis_config() -> RedisStreamConfig:
    return RedisStreamConfig(
        redis_url="redis://unused",
        stream_prefix="stream",
        status_prefix="status",
        expiry_seconds=10,
        max_stream_length=100,
        minimum_connections=None,
    )


@pytest.mark.parametrize("batched", [False, True])
async def test_events_keep_status_alive_until_stream_becomes_idle(
    batched: bool, redis_config: RedisStreamConfig
) -> None:
    """Advance a fake Redis clock beyond the original metadata TTL, without sleeps."""
    now = 0
    metadata = {
        cid: StreamMetadata(
            correlation_id=cid, stream_type="test"
        ).model_dump_json_fields()
        for cid in ("one", "two")
    }
    expires_at = {f"status:{cid}": 10 for cid in metadata}
    replies: list[str | int] = []
    next_event_id = 0

    def xadd(key: str, *args: object, **kwargs: object) -> None:
        nonlocal next_event_id
        next_event_id += 1
        replies.append(f"{next_event_id}-0")

    def expire(key: str, seconds: int) -> None:
        if expires_at.get(key, 0) > now:
            expires_at[key] = now + seconds
        replies.append(1)

    def hgetall(key: str) -> dict[str, str]:
        return metadata[key.removeprefix("status:")] if expires_at[key] > now else {}

    pipe = AsyncMock()

    def enter() -> AsyncMock:
        replies.clear()
        return pipe

    pipe.__aenter__.side_effect = enter
    pipe.xadd.side_effect = xadd
    pipe.expire.side_effect = expire
    pipe.execute.side_effect = lambda: list(replies)
    client = Mock()
    client.pipeline.return_value = pipe
    client.hgetall = AsyncMock(side_effect=hgetall)
    service = RedisStreamService(redis_config, client)

    now = 8
    if batched:
        result = await service.push_event_batch(
            [Event(correlation_id=cid, event_data="partial") for cid in metadata]
        )
        assert result == {"one": "1-0", "two": "2-0"}
    else:
        for cid in metadata:
            await service.push_event(cid, "partial")

    now = 11
    for cid in metadata:
        assert await service.get_stream_metadata(cid) is not None

    now = 19
    for cid in metadata:
        assert await service.get_stream_metadata(cid) is None


async def test_status_update_does_not_recreate_expired_metadata(
    redis_config: RedisStreamConfig,
) -> None:
    client = AsyncMock()
    client.eval.return_value = 0  # No existing key for the atomic status update.
    orphaned_metadata: dict[str, str] = {}
    client.hset.side_effect = lambda key, mapping: orphaned_metadata.update(mapping)
    service = RedisStreamService(redis_config, client)

    await service.update_stream_status("expired", StreamStatus.COMPLETED)

    assert orphaned_metadata == {}
