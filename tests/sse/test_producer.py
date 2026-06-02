from collections.abc import Iterator
from unittest import mock

import pytest

from private_gpt.chat.extensions.citation import ZylonCitation
from private_gpt.components.chunk.models import Chunk
from private_gpt.components.engines.citations.types import Citation
from private_gpt.events.models import (
    PingEvent,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    TextDelta,
)
from private_gpt.events.sse.sse_manager import SSEStreamManager
from private_gpt.events.sse.sse_producer import SSEProducer
from private_gpt.server.ingest.model import IngestedDoc


@pytest.fixture
def manager() -> SSEStreamManager:
    return SSEStreamManager()


@pytest.fixture
def producer(manager: SSEStreamManager) -> SSEProducer:
    return SSEProducer(manager=manager, model_name="test-model")


@pytest.fixture
def mock_chunks() -> list[Chunk]:
    return [
        Chunk(
            object="context.chunk",
            text="First chunk text",
            content_type="text",
            score=0.0,
            document=IngestedDoc(
                object="ingest.document",
                artifact="doc1",
                doc_metadata=None,
            ),
        ),
        Chunk(
            object="context.chunk",
            text="Second chunk text",
            content_type="text",
            score=0.0,
            document=IngestedDoc(
                object="ingest.document",
                artifact="doc2",
                doc_metadata=None,
            ),
        ),
    ]


@pytest.fixture
def mock_citation() -> Citation:
    citation = mock.MagicMock(spec=Citation)
    citation.text = "Sample citation text"
    citation.value = {"id": "cite123", "artifact_id": "art123", "source_id": "src123"}
    citation.doc_id = "doc123"
    return citation


def test_producer_initialization() -> None:
    producer = SSEProducer("test-model")
    assert producer._model_name == "test-model"
    assert producer._input_token_count == 0
    assert producer._output_token_count == 0
    assert producer._content_block_block_id is not None


def test_send_ping(manager: SSEStreamManager, producer: SSEProducer) -> None:
    producer.send_ping()

    # Get event from queue
    event = manager._queue.get_nowait()
    assert isinstance(event, PingEvent)


def test_content_block(manager: SSEStreamManager, producer: SSEProducer) -> None:
    collected_events = []

    old_block_id = producer._content_block_block_id

    with producer.content_block() as block_start:
        collected_events.append(manager._queue.get_nowait())  # Block start
        assert block_start.block_id is not None

    # Collect more events
    while not manager._queue.empty():
        collected_events.append(manager._queue.get_nowait())

    # Check block events
    assert len(collected_events) == 2
    assert isinstance(collected_events[0], RawContentBlockStartEvent)
    assert isinstance(collected_events[1], RawContentBlockStopEvent)
    assert collected_events[0].block_id == old_block_id
    assert collected_events[1].block_id == old_block_id

    # Check content block index is incremented
    assert producer._content_block_block_id != old_block_id


def test_process_content_blocks(
    manager: SSEStreamManager, producer: SSEProducer, mock_citation: Citation
) -> None:
    def block_generator() -> Iterator[RawContentBlockDeltaEvent | Exception | None]:
        block_id = "0"
        yield RawContentBlockDeltaEvent(
            block_id=block_id, delta=TextDelta(text="Hello")
        )
        yield RawContentBlockDeltaEvent(
            block_id=block_id,
            delta=TextDelta(
                text=" world", citations=[ZylonCitation.from_citation(mock_citation)]
            ),
        )
        yield None  # Should be skipped
        yield RawContentBlockDeltaEvent(block_id=block_id, delta=TextDelta(text="!"))

    # Process blocks
    producer.process_content_blocks(block_generator())

    # Collect events
    events = []
    while not manager._queue.empty():
        events.append(manager._queue.get_nowait())

    assert len(events) == 3
    assert all(isinstance(e, RawContentBlockDeltaEvent) for e in events)
    assert events[0].delta.text == "Hello"
    assert events[1].delta.text == " world"
    assert events[1].delta.citations is not None
    assert len(events[1].delta.citations) == 1
    assert events[2].delta.text == "!"


def test_process_content_blocks_with_exception(
    manager: SSEStreamManager, producer: SSEProducer
) -> None:
    def block_generator() -> Iterator[RawContentBlockDeltaEvent | Exception | None]:
        block_id = "0"
        yield RawContentBlockDeltaEvent(
            block_id=block_id, delta=TextDelta(text="Hello")
        )
        yield ValueError("Test error")  # Should be raised
        yield RawContentBlockDeltaEvent(
            block_id=block_id, delta=TextDelta(text="!")
        )  # Won't be reached

    # Process blocks should raise
    with pytest.raises(ValueError, match="Test error"):
        producer.process_content_blocks(block_generator())

    # Check only first event was sent
    events = []
    while not manager._queue.empty():
        events.append(manager._queue.get_nowait())

    assert len(events) == 1
    assert isinstance(events[0], RawContentBlockDeltaEvent)
    assert events[0].delta.text == "Hello"


def test_process_sources(
    manager: SSEStreamManager, producer: SSEProducer, mock_chunks: list[Chunk]
) -> None:
    # Create content block start
    block_id = "0"
    block_start = RawContentBlockStartEvent.from_text(block_id=block_id)

    # Process sources
    producer.process_sources(block_start, mock_chunks)

    # Collect events
    events = []
    while not manager._queue.empty():
        events.append(manager._queue.get_nowait())

    assert len(events) == 1
    assert all(isinstance(e, RawContentBlockDeltaEvent) for e in events)
    assert all(e.delta.type == "source_delta" for e in events)
    assert len(events[0].delta.sources) == 2
    assert events[0].delta.sources[0].document.artifact == "doc1"
    assert events[0].delta.sources[1].document.artifact == "doc2"


def test_set_end_message(manager: SSEStreamManager, producer: SSEProducer):
    # Set end message
    producer.set_end_message("stop")

    # Get event
    event = manager._queue.get_nowait()

    assert isinstance(event, RawMessageDeltaEvent)
    assert event.delta.stop_reason == "stop"
    assert event.usage.output_tokens is None  # No tokens counted

    # Set token count and try again
    producer._output_token_count = 42
    producer.set_end_message()

    event = manager._queue.get()
    assert event.delta.stop_reason == "end_turn"  # Default value
    assert event.usage.output_tokens == 42
