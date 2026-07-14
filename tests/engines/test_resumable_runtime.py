import asyncio

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
    chat_scheduler.cancel.assert_awaited_once_with("execution-cancel")
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
