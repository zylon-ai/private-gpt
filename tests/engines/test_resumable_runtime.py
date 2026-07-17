import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from private_gpt.components.engines.chat.async_chat_engine import (
    IterationCheckpointPayload,
)
from private_gpt.components.engines.chat.checkpoint_store import (
    ChatCheckpoint,
    InMemoryChatCheckpointStore,
)
from private_gpt.components.engines.chat.event_broker import (
    InMemoryEngineEventBroker,
)
from private_gpt.components.engines.chat.event_channel import BrokerEventChannel
from private_gpt.components.tools.remote_execution import ToolExecutionResponse
from private_gpt.events.models import PingEvent, TextBlock


def _resumable_runner(
    checkpoint_store: InMemoryChatCheckpointStore,
    chat_scheduler: MagicMock,
    *,
    settings: MagicMock | None = None,
):
    from private_gpt.components.engines.chat.resumable_runner import ResumableChatRunner

    checkpoint_factory = MagicMock()
    checkpoint_factory.get.return_value = checkpoint_store
    event_factory = MagicMock()
    event_factory.get.return_value = InMemoryEngineEventBroker()
    scheduler_factory = MagicMock()
    scheduler_factory.get.return_value = chat_scheduler
    tool_scheduler_factory = MagicMock()
    tool_scheduler_factory.get.return_value = MagicMock()
    return ResumableChatRunner(
        settings=settings or MagicMock(),
        checkpoint_store_factory=checkpoint_factory,
        event_broker_factory=event_factory,
        scheduler_factory=scheduler_factory,
        tool_scheduler_factory=tool_scheduler_factory,
    )


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


@pytest.mark.asyncio
async def test_runner_executes_duplicate_start_only_once() -> None:
    from private_gpt.components.chat.models.chat_config_models import (
        ResolvedChatRequest,
    )
    from private_gpt.components.engines.chat.models.chat_state import ChatStatus

    checkpoint_store = InMemoryChatCheckpointStore()
    chat_scheduler = MagicMock()
    runner = _resumable_runner(checkpoint_store, chat_scheduler)
    engine = MagicMock()
    completed_state = MagicMock()
    completed_state.output.status = ChatStatus.COMPLETED
    engine.execute = AsyncMock(return_value=completed_state)
    request_data = ResolvedChatRequest(messages=[]).model_dump(mode="json")

    await asyncio.gather(
        *(
            runner.start(
                engine=engine,
                execution_id="execution-start-once",
                request_data=request_data,
                stream_type="chat_completion",
                metadata={},
            )
            for _ in range(10)
        )
    )

    assert engine.execute.await_count == 1


def test_duplicate_tool_call_ids_are_rejected_but_names_may_repeat() -> None:
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.tools import ToolSelection

    from private_gpt.components.engines.chat.async_chat_engine import AsyncChatEngine

    duplicate_ids = ChatMessage(
        role="assistant",
        additional_kwargs={
            "tool_calls": [
                ToolSelection(tool_id="duplicate", tool_name="search", tool_kwargs={}),
                ToolSelection(
                    tool_id="duplicate", tool_name="database", tool_kwargs={}
                ),
            ]
        },
    )
    same_name = ChatMessage(
        role="assistant",
        additional_kwargs={
            "tool_calls": [
                ToolSelection(tool_id="tool-1", tool_name="search", tool_kwargs={}),
                ToolSelection(tool_id="tool-2", tool_name="search", tool_kwargs={}),
            ]
        },
    )

    with pytest.raises(RuntimeError, match="Duplicate tool call ID"):
        AsyncChatEngine._validate_unique_tool_call_ids(duplicate_ids)
    AsyncChatEngine._validate_unique_tool_call_ids(same_name)


@pytest.mark.asyncio
async def test_result_envelope_cannot_write_a_different_tool_id() -> None:
    service = InMemoryChatCheckpointStore()
    await service.save(
        ChatCheckpoint(
            correlation_id="execution-envelope",
            request_data={},
            stream_type="chat_completion",
            metadata={},
            iteration=0,
            checkpoint="tools",
            checkpoint_payload=IterationCheckpointPayload(
                pending_async_tools={"expected-tool": "task-1"}
            ),
        )
    )
    malformed = _tool_response("wrong-tool").model_dump(mode="json")

    await service.record_result("execution-envelope", "expected-tool", malformed)

    result = (await service.get_results("execution-envelope"))["expected-tool"]
    assert result.tool_id == "expected-tool"
    assert result.tool_message.additional_kwargs["tool_call_id"] == "expected-tool"


@pytest.mark.asyncio
async def test_memory_event_broker_preserves_publish_order_and_finishes() -> None:
    broker = InMemoryEngineEventBroker()
    channel = BrokerEventChannel(broker, "execution-1")
    channel.emit(PingEvent())
    channel.emit(PingEvent())
    await channel.close()
    await broker.finish("execution-1")

    events = [event async for event in broker.listen("execution-1")]

    assert [event.type for event in events] == ["ping", "ping"]


@pytest.mark.asyncio
async def test_memory_event_broker_waits_for_late_publication() -> None:
    broker = InMemoryEngineEventBroker()

    async def collect():
        return [event async for event in broker.listen("execution-2")]

    collector = asyncio.create_task(collect())
    await asyncio.sleep(0)
    await broker.publish("execution-2", PingEvent())
    await broker.finish("execution-2")

    assert [event.type for event in await collector] == ["ping"]


@pytest.mark.asyncio
async def test_memory_iteration_state_aggregates_once_and_cleans_up() -> None:
    service = InMemoryChatCheckpointStore()
    response = ToolExecutionResponse(
        tool_name="echo",
        tool_id="tool-1",
        result_content=[TextBlock(text="ok")],
        tool_message={
            "role": "tool",
            "content": "ok",
            "additional_kwargs": {"tool_call_id": "tool-1"},
        },
    )
    await service.save(
        ChatCheckpoint(
            correlation_id="execution-3",
            request_data={},
            stream_type="chat_completion",
            metadata={},
            iteration=0,
            checkpoint="tools",
            checkpoint_payload=IterationCheckpointPayload(
                pending_async_tools={"tool-1": "job-1"}
            ),
            checkpoint_id="checkpoint-1",
        )
    )

    ready = await service.record_result(
        "execution-3", "tool-1", response.model_dump(mode="json")
    )

    assert ready == {"tool-1": response}
    assert await service.claim_resume("execution-3") is True
    assert await service.claim_resume("execution-3") is False
    await service.cleanup("execution-3")
    assert await service.load("execution-3") is None


@pytest.mark.asyncio
async def test_runner_waits_for_all_parallel_tools_before_resuming() -> None:
    checkpoint_store = InMemoryChatCheckpointStore()
    await checkpoint_store.save(
        ChatCheckpoint(
            correlation_id="execution-parallel",
            request_data={},
            stream_type="chat_completion",
            metadata={},
            iteration=0,
            checkpoint="tools",
            checkpoint_payload=IterationCheckpointPayload(
                pending_async_tools={"tool-1": "job-1", "tool-2": "job-2"}
            ),
            checkpoint_id="parallel-checkpoint",
        )
    )
    chat_scheduler = MagicMock()
    chat_scheduler.resume = AsyncMock()
    runner = _resumable_runner(checkpoint_store, chat_scheduler)

    await asyncio.gather(
        runner.callback(
            execution_id="execution-parallel",
            tool_id="tool-1",
            result=_tool_response("tool-1").model_dump(mode="json"),
        ),
        runner.callback(
            execution_id="execution-parallel",
            tool_id="tool-2",
            result=_tool_response("tool-2").model_dump(mode="json"),
        ),
    )
    chat_scheduler.resume.assert_awaited_once_with(
        execution_id="execution-parallel",
        checkpoint_id="parallel-checkpoint",
    )

    await runner.callback(
        execution_id="execution-parallel",
        tool_id="tool-2",
        result=_tool_response("tool-2").model_dump(mode="json"),
    )
    assert chat_scheduler.resume.await_count == 1


@pytest.mark.asyncio
async def test_runner_schedules_one_timeout_timer_per_pending_tool() -> None:
    from llama_index.core.base.llms.types import ChatMessage
    from llama_index.core.tools import ToolSelection

    from private_gpt.components.chat.models.chat_config_models import (
        ResolvedChatRequest,
    )
    from private_gpt.components.engines.chat.models.chat_state import ChatStatus

    checkpoint_store = InMemoryChatCheckpointStore()
    chat_scheduler = MagicMock()
    chat_scheduler.tool_timeout = AsyncMock()
    settings = MagicMock()
    settings.scheduler.chat.callback_timeout_seconds = 45
    runner = _resumable_runner(checkpoint_store, chat_scheduler, settings=settings)
    state = MagicMock()
    state.output.status = ChatStatus.WAITING
    state.output.pause_type = "tools"
    state.output.pending_async_tools = {
        "tool-1": "celery-task-1",
        "tool-2": "celery-task-2",
    }
    state.output.pending_external_tool_calls = []
    request = ResolvedChatRequest(
        messages=[
            ChatMessage(
                role="assistant",
                content=None,
                additional_kwargs={
                    "tool_calls": [
                        ToolSelection(
                            tool_id="tool-1",
                            tool_name="first_tool",
                            tool_kwargs={},
                        ),
                        ToolSelection(
                            tool_id="tool-2",
                            tool_name="second_tool",
                            tool_kwargs={},
                        ),
                    ]
                },
            )
        ]
    )
    state.input.request.model_dump.return_value = request.model_dump(mode="json")
    state.input.context_stack.checkpoint_dump.return_value = {}
    state.runtime.iteration = 1
    state.runtime.next_block_count = 3
    state.runtime.total_input_tokens = 0
    state.runtime.total_output_tokens = 0
    state.runtime.has_input_usage = False
    state.runtime.has_output_usage = False

    await runner._handle_state(
        execution_id="execution-timers",
        state=state,
        stream_type="chat_completion",
        metadata={},
    )

    assert chat_scheduler.tool_timeout.await_count == 2
    calls = {
        call.kwargs["tool_id"]: call.kwargs
        for call in chat_scheduler.tool_timeout.await_args_list
    }
    assert calls["tool-1"]["task_id"] == "celery-task-1"
    assert calls["tool-2"]["task_id"] == "celery-task-2"
    assert calls["tool-1"]["delay_seconds"] == 45
    assert calls["tool-2"]["delay_seconds"] == 45
    assert calls["tool-1"]["tool_name"] == "first_tool"
    assert calls["tool-2"]["tool_name"] == "second_tool"
    assert calls["tool-1"]["checkpoint_id"] == calls["tool-2"]["checkpoint_id"]


@pytest.mark.asyncio
async def test_runner_releases_resume_claim_when_dispatch_fails() -> None:
    checkpoint_store = InMemoryChatCheckpointStore()
    await checkpoint_store.save(
        ChatCheckpoint(
            correlation_id="execution-retry",
            request_data={},
            stream_type="chat_completion",
            metadata={},
            iteration=0,
            checkpoint="tools",
            checkpoint_payload=IterationCheckpointPayload(
                pending_async_tools={"tool-1": "job-1"}
            ),
            checkpoint_id="retry-checkpoint",
        )
    )
    chat_scheduler = MagicMock()
    chat_scheduler.resume = AsyncMock(
        side_effect=[RuntimeError("ARQ unavailable"), None]
    )
    runner = _resumable_runner(checkpoint_store, chat_scheduler)
    result = _tool_response("tool-1").model_dump(mode="json")

    with pytest.raises(RuntimeError, match="ARQ unavailable"):
        await runner.callback(
            execution_id="execution-retry", tool_id="tool-1", result=result
        )

    await runner.callback(
        execution_id="execution-retry", tool_id="tool-1", result=result
    )

    assert chat_scheduler.resume.await_count == 2


@pytest.mark.asyncio
async def test_memory_iteration_state_preserves_result_received_before_checkpoint() -> (
    None
):
    service = InMemoryChatCheckpointStore()
    response = ToolExecutionResponse(
        tool_name="echo",
        tool_id="tool-early",
        result_content=[TextBlock(text="ok")],
        tool_message={
            "role": "tool",
            "content": "ok",
            "additional_kwargs": {"tool_call_id": "tool-early"},
        },
    )

    assert (
        await service.record_result(
            "execution-early", "tool-early", response.model_dump(mode="json")
        )
        is None
    )
    await service.save(
        ChatCheckpoint(
            correlation_id="execution-early",
            request_data={},
            stream_type="chat_completion",
            metadata={},
            iteration=0,
            checkpoint="tools",
            checkpoint_payload=IterationCheckpointPayload(
                pending_async_tools={"tool-early": "job-early"}
            ),
        )
    )

    assert await service.get_results("execution-early") == {"tool-early": response}


@pytest.mark.asyncio
async def test_memory_iteration_state_preserves_next_iteration_result() -> None:
    service = InMemoryChatCheckpointStore()
    await service.save(
        ChatCheckpoint(
            correlation_id="execution-unknown",
            request_data={},
            stream_type="chat_completion",
            metadata={},
            iteration=0,
            checkpoint="tools",
            checkpoint_payload=IterationCheckpointPayload(
                pending_async_tools={"tool-expected": "job-expected"}
            ),
        )
    )
    response = ToolExecutionResponse(
        tool_name="echo",
        tool_id="tool-unknown",
        result_content=[TextBlock(text="ok")],
        tool_message={
            "role": "tool",
            "content": "ok",
            "additional_kwargs": {"tool_call_id": "tool-unknown"},
        },
    )

    assert (
        await service.record_result(
            "execution-unknown", "tool-unknown", response.model_dump(mode="json")
        )
        is None
    )
    assert await service.get_results("execution-unknown") == {"tool-unknown": response}

    await service.save(
        ChatCheckpoint(
            correlation_id="execution-unknown",
            request_data={},
            stream_type="chat_completion",
            metadata={},
            iteration=1,
            checkpoint="tools",
            checkpoint_payload=IterationCheckpointPayload(
                pending_async_tools={"tool-unknown": "job-next"}
            ),
        )
    )

    assert await service.get_results("execution-unknown") == {"tool-unknown": response}


@pytest.mark.asyncio
async def test_terminal_execution_rejects_new_checkpoints_and_late_results() -> None:
    service = InMemoryChatCheckpointStore()
    assert await service.mark_terminal("execution-terminal", "cancelled") is True

    saved = await service.save(
        ChatCheckpoint(
            correlation_id="execution-terminal",
            request_data={},
            stream_type="chat_completion",
            metadata={},
            iteration=1,
            checkpoint="tools",
        )
    )
    recorded = await service.record_result(
        "execution-terminal",
        "tool-late",
        _tool_response("tool-late").model_dump(mode="json"),
    )

    assert saved is False
    assert recorded is None
    assert await service.load("execution-terminal") is None
    assert await service.get_results("execution-terminal") == {}


@pytest.mark.asyncio
async def test_cancelled_chat_cannot_resurrect_waiting_checkpoint() -> None:
    from private_gpt.components.engines.chat.models.chat_state import ChatStatus

    checkpoint_store = InMemoryChatCheckpointStore()
    await checkpoint_store.mark_terminal("execution-no-resurrection", "cancelled")
    chat_scheduler = MagicMock()
    chat_scheduler.tool_timeout = AsyncMock()
    settings = MagicMock()
    settings.scheduler.chat.callback_timeout_seconds = 30
    runner = _resumable_runner(checkpoint_store, chat_scheduler, settings=settings)
    runner._tool_scheduler.cancel_task = AsyncMock(return_value=True)
    state = MagicMock()
    state.output.status = ChatStatus.WAITING
    state.output.pause_type = "tools"
    state.output.pending_async_tools = {
        "tool-1": "celery-task-1",
        "tool-2": "celery-task-2",
    }
    state.output.pending_external_tool_calls = []
    state.input.request.model_dump.return_value = {"messages": []}
    state.input.context_stack.checkpoint_dump.return_value = {}
    state.runtime.iteration = 1
    state.runtime.next_block_count = 0
    state.runtime.total_input_tokens = 0
    state.runtime.total_output_tokens = 0
    state.runtime.has_input_usage = False
    state.runtime.has_output_usage = False

    await runner._handle_state(
        execution_id="execution-no-resurrection",
        state=state,
        stream_type="chat_completion",
        metadata={},
    )

    assert runner._tool_scheduler.cancel_task.await_count == 2
    runner._tool_scheduler.cancel_task.assert_any_await(task_id="celery-task-1")
    runner._tool_scheduler.cancel_task.assert_any_await(task_id="celery-task-2")
    chat_scheduler.tool_timeout.assert_not_awaited()
    assert await checkpoint_store.load("execution-no-resurrection") is None


@pytest.mark.asyncio
async def test_runner_cancel_revokes_pending_tool_tasks() -> None:
    from unittest.mock import AsyncMock, MagicMock

    from private_gpt.components.engines.chat.resumable_runner import ResumableChatRunner

    checkpoint_store = InMemoryChatCheckpointStore()
    await checkpoint_store.save(
        ChatCheckpoint(
            correlation_id="execution-cancel",
            request_data={},
            stream_type="chat_completion",
            metadata={},
            iteration=1,
            checkpoint="tools",
            checkpoint_payload=IterationCheckpointPayload(
                pending_async_tools={
                    "tool-1": "celery-task-1",
                    "tool-2": "celery-task-2",
                }
            ),
            checkpoint_id="cancel-checkpoint",
        )
    )
    event_broker = InMemoryEngineEventBroker()
    chat_scheduler = MagicMock()
    chat_scheduler.cancel = AsyncMock(return_value=False)
    tool_scheduler = MagicMock()
    tool_scheduler.cancel_task = AsyncMock(return_value=True)

    checkpoint_factory = MagicMock()
    checkpoint_factory.get.return_value = checkpoint_store
    event_factory = MagicMock()
    event_factory.get.return_value = event_broker
    scheduler_factory = MagicMock()
    scheduler_factory.get.return_value = chat_scheduler
    tool_scheduler_factory = MagicMock()
    tool_scheduler_factory.get.return_value = tool_scheduler

    runner = ResumableChatRunner(
        settings=MagicMock(),
        checkpoint_store_factory=checkpoint_factory,
        event_broker_factory=event_factory,
        scheduler_factory=scheduler_factory,
        tool_scheduler_factory=tool_scheduler_factory,
    )

    cancelled = await runner.cancel(execution_id="execution-cancel")

    assert cancelled is True
    assert tool_scheduler.cancel_task.await_count == 2
    tool_scheduler.cancel_task.assert_any_await(task_id="celery-task-1")
    tool_scheduler.cancel_task.assert_any_await(task_id="celery-task-2")
    chat_scheduler.cancel.assert_awaited_once_with(
        "execution-cancel",
        checkpoint_id="cancel-checkpoint",
        tool_ids=("tool-1", "tool-2"),
    )
    assert await checkpoint_store.load("execution-cancel") is None


def test_runner_restores_additional_kwargs_after_serialization() -> None:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from llama_index.core.tools import ToolSelection

    from private_gpt.components.chat.models.chat_config_models import (
        ResolvedChatRequest,
    )
    from private_gpt.components.engines.chat.resumable_runner import (
        ResumableChatRunner,
    )
    from private_gpt.events.models import DocumentBlock, ThinkingBlock

    document = DocumentBlock(
        source=DocumentBlock.PlainTextSource(
            data="Document text",
            media_type="text/plain",
        ),
        title="Test document",
    )
    thinking = ThinkingBlock(thinking="Reasoning", signature="signature")
    tool_call = ToolSelection(
        tool_id="tool-1",
        tool_name="lookup",
        tool_kwargs={"query": "test"},
    )
    request = ResolvedChatRequest(
        messages=[
            ChatMessage(
                role=MessageRole.USER,
                content="Summarize this document",
                additional_kwargs={
                    "document": [document],
                    "thinking": [thinking],
                    "tool_calls": [tool_call],
                },
            )
        ]
    )

    restored = ResumableChatRunner._request(request.model_dump(mode="json"))
    additional_kwargs = restored.messages[0].additional_kwargs

    assert additional_kwargs["document"] == [document]
    assert additional_kwargs["thinking"] == [thinking]
    assert additional_kwargs["tool_calls"] == [tool_call]
    assert isinstance(additional_kwargs["document"][0], DocumentBlock)
    assert isinstance(additional_kwargs["thinking"][0], ThinkingBlock)
    assert isinstance(additional_kwargs["tool_calls"][0], ToolSelection)


def test_runner_orders_results_by_pending_tool_order() -> None:
    from private_gpt.components.engines.chat.resumable_runner import ResumableChatRunner

    results = {
        "tool-2": ToolExecutionResponse(
            tool_name="second",
            tool_id="tool-2",
            result_content=[TextBlock(text="second")],
            tool_message={
                "role": "tool",
                "content": "second",
                "additional_kwargs": {"tool_call_id": "tool-2"},
            },
        ),
        "tool-1": ToolExecutionResponse(
            tool_name="first",
            tool_id="tool-1",
            result_content=[TextBlock(text="first")],
            tool_message={
                "role": "tool",
                "content": "first",
                "additional_kwargs": {"tool_call_id": "tool-1"},
            },
        ),
    }
    checkpoint = ChatCheckpoint(
        correlation_id="execution-order",
        request_data={},
        stream_type="chat_completion",
        metadata={},
        iteration=0,
        checkpoint="tools",
        checkpoint_payload=IterationCheckpointPayload(
            pending_async_tools={"tool-1": "job-1", "tool-2": "job-2"}
        ),
    )

    ordered = ResumableChatRunner._ordered_results(checkpoint, results)

    assert [response.tool_id for response in ordered] == ["tool-1", "tool-2"]


def test_resolved_chat_request_is_json_roundtrip_serializable() -> None:
    import json

    from llama_index.core.base.llms.types import ChatMessage, MessageRole
    from llama_index.core.tools import ToolSelection

    from private_gpt.chat.schema_models import create_model_from_json_schema
    from private_gpt.components.chat.models.chat_config_models import (
        CitationConfig,
        CondensationConfig,
        ResolvedChatRequest,
        ResolvedContextConfig,
        ResolvedSystemConfig,
        ResolvedToolConfig,
        ResponseFormatConfig,
        ThinkingConfig,
        ToolSpec,
    )
    from private_gpt.components.engines.chat.resumable_runner import (
        ResumableChatRunner,
    )
    from private_gpt.events.models import DocumentBlock, ThinkingBlock

    document = DocumentBlock(
        source=DocumentBlock.PlainTextSource(
            data="Document text",
            media_type="text/plain",
        ),
        title="Test document",
    )
    thinking = ThinkingBlock(thinking="Reasoning", signature="signature")
    tool_call = ToolSelection(
        tool_id="tool-1",
        tool_name="lookup",
        tool_kwargs={"query": "test"},
    )
    output_schema = {
        "title": "Profile",
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    output_cls = create_model_from_json_schema(output_schema)
    request = ResolvedChatRequest(
        stream=True,
        messages=[
            ChatMessage(
                role=MessageRole.USER,
                content="Create a profile",
                additional_kwargs={
                    "document": [document],
                    "thinking": [thinking],
                    "tool_calls": [tool_call],
                },
            )
        ],
        system=ResolvedSystemConfig(prompt="System prompt", model="default"),
        tool_config=ResolvedToolConfig(
            tools=[
                ToolSpec.from_defaults(
                    name="lookup",
                    type="lookup",
                    runtime="server",
                    input_schema={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                )
            ],
            tool_choices=["lookup"],
            allow_parallel_tool_calls=False,
        ),
        context=ResolvedContextConfig(
            add_context_to_system_prompt=True,
            deduplicate_context_in_history=True,
            maximum_context_length=2048,
            correlation_id="correlation-1",
            user_id="user-1",
            container="container-1",
            maximum_loaded_skills=2,
        ),
        condensation=CondensationConfig(enabled=False),
        citation=CitationConfig(
            enabled=True,
            force_to_return_citations=True,
            return_missing_citations=True,
        ),
        thinking=ThinkingConfig(enabled=True, type="high"),
        response_format=ResponseFormatConfig(output_cls=output_cls),
        sampling_params={
            "temperature": 0.2,
            "max_tokens": 512,
            "stop": ["done"],
        },
    )

    serialized = request.model_dump(mode="json")
    encoded = json.dumps(serialized)
    restored = ResumableChatRunner._request(json.loads(encoded))

    assert set(serialized) == set(ResolvedChatRequest.model_fields)
    assert restored.model_dump(mode="json") == serialized


def test_checkpoint_context_stack_roundtrip_restores_durable_layers_and_tools() -> None:
    from llama_index.core.base.llms.types import ChatMessage, MessageRole

    from private_gpt.components.chat.models.chat_config_models import (
        ResolvedChatRequest,
        ResolvedSystemConfig,
        ResolvedToolConfig,
        ToolExecutionMetadata,
        ToolSpec,
    )
    from private_gpt.components.context.models.context_layer import (
        RuntimeInstructionsLayer,
        ToolDefinitionsLayer,
    )
    from private_gpt.components.context.models.context_stack import ContextStack
    from private_gpt.components.engines.chat.resumable_runner import ResumableChatRunner

    tool = ToolSpec.from_defaults(
        name="lookup",
        type="lookup",
        runtime="server",
        input_schema={"type": "object", "properties": {}},
        execution_metadata=ToolExecutionMetadata(
            rebuild_callable="tests.engines.test_async_chat_engine:_rebuild_server_tool",
            rebuild_kwargs={"name": "lookup"},
        ),
    )
    request = ResolvedChatRequest(
        messages=[ChatMessage(role=MessageRole.USER, content="hello")],
        system=ResolvedSystemConfig(model="default"),
        tool_config=ResolvedToolConfig(tools=[tool]),
    )
    stack = ContextStack(
        layers=[
            RuntimeInstructionsLayer(text="keep me", source="runtime"),
            ToolDefinitionsLayer(tools=[tool], source="mcp"),
        ]
    )
    checkpoint = ChatCheckpoint(
        correlation_id="execution-context",
        request_data=request.model_dump(mode="json"),
        context_stack_data=stack.checkpoint_dump(),
        stream_type="chat_completion",
        metadata={},
        iteration=1,
    )

    restored_checkpoint = ChatCheckpoint.model_validate_json(
        checkpoint.model_dump_json()
    )
    restored = ResumableChatRunner._context_stack(
        restored_checkpoint, restored_checkpoint.request_data
    )

    assert restored.layers[0] == RuntimeInstructionsLayer(
        text="keep me", source="runtime"
    )
    assert len(restored.all_tools()) == 1
    assert restored.all_tools()[0].name == "lookup"
    assert restored.all_tools()[0].execution_metadata is not None
    assert any(layer.source == "mcp" for layer in restored.layers)


def test_checkpoint_context_stack_rejects_non_resumable_server_tool() -> None:
    from private_gpt.components.chat.models.chat_config_models import ToolSpec
    from private_gpt.components.context.errors import NonResumableToolError
    from private_gpt.components.context.models.context_layer import ToolDefinitionsLayer
    from private_gpt.components.context.models.context_stack import ContextStack

    stack = ContextStack(
        layers=[
            ToolDefinitionsLayer(
                tools=[
                    ToolSpec.from_defaults(
                        name="ephemeral",
                        runtime="server",
                        input_schema={"type": "object", "properties": {}},
                    )
                ],
                source="test",
            )
        ]
    )

    with pytest.raises(NonResumableToolError, match="ephemeral"):
        stack.checkpoint_dump()
