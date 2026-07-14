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
