import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from llama_index.core.tools import ToolSelection

from private_gpt.components.chat.models.chat_config_models import (
    ResolvedChatRequest,
    ResolvedToolConfig,
    ToolSpec,
)
from private_gpt.components.context.models.context_layer import ToolDefinitionsLayer
from private_gpt.components.context.models.context_stack import ContextStack
from private_gpt.components.engines.chat.async_chat_engine import AsyncChatEngine
from private_gpt.components.engines.chat.checkpoint_store import ChatCheckpoint
from private_gpt.components.engines.chat.models.chat_state import (
    ChatInputState,
    ChatOutputState,
    ChatRuntimeState,
    ChatState,
)
from private_gpt.components.engines.chat.models.execution_hooks import ExecutionHooks
from private_gpt.components.engines.chat.resumable_runner import ResumableChatRunner
from private_gpt.components.engines.chat.utils.tool_utils import (
    apply_mcp_token_refreshes,
)
from private_gpt.components.tools.remote_execution import ToolExecutionResponse
from private_gpt.components.tools.tool_execution_outcome import ToolExecutionSuccess
from private_gpt.events.models import (
    McpTokensRefreshedEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TextBlock,
    ToolResultBlock,
)
from private_gpt.server.mcp.config import McpServerConfig
from private_gpt.server.mcp.mcp_service import McpToolDefinition, mcp_tool_to_spec


def _event() -> McpTokensRefreshedEvent:
    return McpTokensRefreshedEvent(
        name="tools",
        url="https://mcp.example.com",
        previous_refresh_token="refresh-before-sentinel",
        authorization_token="access-after-sentinel",
        refresh_token="refresh-after-sentinel",
        metadata={"artifact_id": "artifact-123"},
    )


def _mcp_config(tool: ToolSpec) -> McpServerConfig:
    assert tool.execution_metadata is not None
    config = tool.execution_metadata.rebuild_kwargs["config"]
    assert isinstance(config, McpServerConfig)
    return config


def test_refresh_event_updates_durable_original_tool_config() -> None:
    config = McpServerConfig(
        name="tools",
        url="https://mcp.example.com",
        authorization_token="access-before-sentinel",
        refresh_token="refresh-before-sentinel",
        client_id="client-id",
    )
    tool = mcp_tool_to_spec(
        config,
        McpToolDefinition(
            name="lookup",
            description=None,
            input_schema={"type": "object", "properties": {}},
        ),
    )
    input_state = ChatInputState(
        request=ResolvedChatRequest(
            messages=[],
            tool_config=ResolvedToolConfig(tools=[tool]),
        ),
        context_stack=ContextStack(
            layers=[ToolDefinitionsLayer(tools=[tool], source="mcp")]
        ),
    )
    state = ChatState(
        input=input_state,
        original_input=input_state.model_copy(deep=True),
        runtime=ChatRuntimeState(),
        output=ChatOutputState(),
    )

    apply_mcp_token_refreshes(state, [_event()])

    assert state.original_input is not None
    checkpoint = ChatCheckpoint(
        correlation_id="execution-with-mcp-refresh",
        request_data={},
        original_input_data=ResumableChatRunner._dump_original_input(
            state.original_input
        ),
        stream_type="chat_completion",
        metadata={},
        iteration=1,
    )
    restored_original = ResumableChatRunner._original_input(checkpoint)
    assert restored_original is not None
    for persisted_tool in (
        *state.input.context_stack.all_tools(),
        *restored_original.context_stack.all_tools(),
    ):
        persisted_config = _mcp_config(persisted_tool)
        assert persisted_config.authorization_token == "access-after-sentinel"
        assert persisted_config.refresh_token == "refresh-after-sentinel"


@pytest.mark.asyncio
async def test_inline_tool_execution_emits_internal_event_before_tool_result() -> None:
    engine = object.__new__(AsyncChatEngine)
    scheduler = MagicMock()
    scheduler.is_async = False
    scheduler.execute = AsyncMock(
        return_value=ToolExecutionResponse(
            tool_name="lookup",
            tool_id="tool-1",
            outcome=ToolExecutionSuccess(content=[TextBlock(text="normal result")]),
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
