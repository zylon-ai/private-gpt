import json
from collections.abc import AsyncGenerator

import pytest

from private_gpt.events.event_serializer import StreamingEventHandler
from private_gpt.events.interceptors.filter_zylon_event_interceptor import (
    FilterZylonEventInterceptor,
)
from private_gpt.events.models import Event, McpTokensRefreshedEvent


def _event() -> McpTokensRefreshedEvent:
    return McpTokensRefreshedEvent(
        name="tools",
        url="https://mcp.example.com",
        authorization_token="access-after-sentinel",
        refresh_token="refresh-after-sentinel",
        metadata={"artifact_id": "artifact-123"},
    )


def test_mcp_tokens_refreshed_event_roundtrips_exactly_and_redacts_strings() -> None:
    event = _event()
    serialized = StreamingEventHandler().serialize(event)

    assert json.loads(serialized) == {
        "type": "mcp_tokens_refreshed",
        "name": "tools",
        "url": "https://mcp.example.com",
        "authorization_token": "access-after-sentinel",
        "refresh_token": "refresh-after-sentinel",
        "_meta": {"artifact_id": "artifact-123"},
    }
    restored = StreamingEventHandler().deserialize(serialized)
    assert restored == event
    assert "tools" in str(event)
    assert "https://mcp.example.com" in repr(event)
    for sentinel in (
        "access-after-sentinel",
        "refresh-after-sentinel",
    ):
        assert sentinel not in str(event)
        assert sentinel not in repr(event)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("response_mode", "expected"),
    [("anthropic", []), ("zylon", ["artifact-123"])],
)
async def test_mcp_tokens_refreshed_event_is_zylon_only(
    response_mode: str, expected: list[str]
) -> None:
    async def source() -> AsyncGenerator[Event, None]:
        yield _event()

    filtered = await FilterZylonEventInterceptor(response_mode).intercept(source())
    events = [event async for event in filtered]

    assert [event.metadata["artifact_id"] for event in events] == expected
