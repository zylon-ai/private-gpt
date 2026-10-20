import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from private_gpt.arq.chat.iteration_state import RedisChatCheckpointStore
from private_gpt.components.tools.remote_execution import ToolExecutionResponse
from private_gpt.events.models import TextBlock


def _tool_response(tool_id: str) -> ToolExecutionResponse:
    return ToolExecutionResponse(
        tool_name=f"tool-{tool_id}",
        tool_id=tool_id,
        result_content=[TextBlock(text=f"result-{tool_id}")],
        tool_message={
            "role": "tool",
            "content": f"result-{tool_id}",
            "additional_kwargs": {"tool_call_id": tool_id},
        },
    )


def _store(stored: dict[str, str]) -> RedisChatCheckpointStore:
    settings = MagicMock()
    settings.scheduler.chat.callback_timeout_seconds = 60
    with (
        patch(
            "private_gpt.arq.chat.iteration_state.get_redis_settings",
            return_value=MagicMock(host="localhost", port=6379),
        ),
        patch("private_gpt.arq.chat.iteration_state.redis.Redis") as redis_cls,
    ):
        redis_cls.return_value.hgetall = AsyncMock(return_value=stored)
        return RedisChatCheckpointStore(settings)


@pytest.mark.asyncio
async def test_get_results_parses_tool_results_off_the_event_loop_thread() -> None:
    loop_thread = threading.get_ident()
    parse_threads: list[int] = []
    expected = {tool_id: _tool_response(tool_id) for tool_id in ("a", "b")}
    stored = {tool_id: value.model_dump_json() for tool_id, value in expected.items()}
    original = ToolExecutionResponse.model_validate_json

    def model_validate_json(value: Any, *args: Any, **kwargs: Any) -> Any:
        parse_threads.append(threading.get_ident())
        return original(value, *args, **kwargs)

    store = _store(stored)
    with patch.object(
        ToolExecutionResponse, "model_validate_json", side_effect=model_validate_json
    ):
        results = await store.get_results("execution")

    assert results == expected
    assert len(parse_threads) == 2
    assert loop_thread not in parse_threads
