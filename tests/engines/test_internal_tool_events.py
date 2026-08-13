import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.tools import ToolSelection

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.engines.chat.async_chat_engine import AsyncChatEngine
from private_gpt.components.engines.chat.models.execution_hooks import ExecutionHooks
from private_gpt.components.tools.remote_execution import ToolExecutionResponse
from private_gpt.events.models import (
    McpTokensRefreshedEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TextBlock,
    ToolResultBlock,
)


def _event() -> McpTokensRefreshedEvent:
    return McpTokensRefreshedEvent(
        artifact_id="artifact-123",
        previous_refresh_token="refresh-before-sentinel",
        authorization_token="access-after-sentinel",
        refresh_token="refresh-after-sentinel",
    )


@pytest.mark.asyncio
async def test_inline_tool_execution_emits_internal_event_before_tool_result() -> None:
    engine = object.__new__(AsyncChatEngine)
    scheduler = MagicMock()
    scheduler.is_async = False
    scheduler.execute = AsyncMock(
        return_value=ToolExecutionResponse(
            tool_name="lookup",
            tool_id="tool-1",
            result_content=[TextBlock(text="normal result")],
            tool_message={
                "role": "tool",
                "content": "normal result",
                "additional_kwargs": {"tool_call_id": "tool-1"},
            },
            internal_events=[_event()],
        )
    )
    engine._tool_scheduler = scheduler
    engine._tool_interceptors = []

    state = MagicMock()
    state.input.request.context.correlation_id = None
    state.input.request.messages = []
    state.model_copy.return_value = state
    run = MagicMock()
    run.state = state
    run.block_count = 0
    run.hooks = ExecutionHooks()

    emitted: list[object] = []
    handler = MagicMock()
    handler.emit.side_effect = emitted.append

    result = await engine._handle_tool_use(
        run=run,
        tool_call=ToolSelection(
            tool_id="raw-tool-id",
            tool_name="lookup",
            tool_kwargs={},
        ),
        tool_specs_by_name={
            "lookup": ToolSpec(
                name="lookup",
                runtime="server",
                input_schema={"type": "object", "properties": {}},
            )
        },
        handler=handler,
        tool_id_map={"raw-tool-id": "tool-1"},
        lock=asyncio.Lock(),
    )

    assert result.status == "executed"
    assert emitted[0] == _event()
    assert isinstance(emitted[1], RawContentBlockStartEvent)
    assert isinstance(emitted[2], RawContentBlockStopEvent)
    assert isinstance(emitted[1].content_block, ToolResultBlock)
    assert emitted[1].content_block.content == [TextBlock(text="normal result")]
