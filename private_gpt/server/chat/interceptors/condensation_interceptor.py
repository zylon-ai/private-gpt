import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
from typing import cast
from uuid import uuid4

from injector import inject, singleton
from llama_index.core.base.llms.types import ChatMessage
from pydantic import BaseModel

from private_gpt.components.chat.processors.chat_history.memory.tldr_processor import (
    CondenseResponse,
    condense_chat_history,
)
from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import (
    InterceptorPhase,
)
from private_gpt.events.models import (
    Event,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    TLDRBlock,
    TLDRDelta,
)
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

# Sentinel used to signal the end of the queue
_SENTINEL: object = object()


def _token_limit_with_buffer(token_limit: int, token_buffer: float) -> int:
    return max(1, int(token_limit * (1 - token_buffer)))


class _CondensationResult(BaseModel):
    chat_history: list[ChatMessage] | None = None
    condensed: bool = False


async def _condensation_producer(
    queue: asyncio.Queue[Event | object | None],
    result: _CondensationResult,
    generator: AsyncIterator[CondenseResponse],
) -> _CondensationResult:
    blocks: dict[str, RawContentBlockStartEvent] = {}

    try:
        async for response in generator:
            if not response.is_condensed:
                continue

            if (
                not blocks
                and response.condense_blocks is not None
                and response.chat_history is None
            ):
                block = RawContentBlockStartEvent(
                    block_id=f"block_{uuid4().hex}",
                    content_block=TLDRBlock(content=[], tldr_side="left"),
                )
                blocks["left"] = block
                await queue.put(block)

            if response.condense_blocks and response.chat_history is not None:
                all_tldr_sides = {
                    content_block.metadata.get("tldr_side", "left")
                    for content_block in response.condense_blocks
                }

                for content_block in response.condense_blocks:
                    tldr_side = content_block.metadata.get("tldr_side", "left")

                    if tldr_side not in blocks:
                        if len(all_tldr_sides) == 1 and "left" in blocks:
                            blocks[tldr_side] = blocks.pop("left")
                        else:
                            block = RawContentBlockStartEvent(
                                block_id=f"block_{uuid4().hex}",
                                content_block=TLDRBlock(
                                    content=[],
                                    tldr_side=tldr_side,  # type: ignore[arg-type]
                                ),
                            )
                            blocks[tldr_side] = block
                            await queue.put(block)

                    block = blocks[tldr_side]
                    await queue.put(
                        RawContentBlockDeltaEvent(
                            index=block.index,
                            block_id=block.block_id,
                            delta=TLDRDelta(
                                tldr=content_block,
                                tldr_side=tldr_side,  # type: ignore[arg-type]
                            ),
                        )
                    )

            if response.chat_history is not None:
                result.chat_history = response.chat_history
                result.condensed = True

    except asyncio.CancelledError:
        # Routine: the client disconnected or the request was aborted.
        logger.debug(
            "TLDR condensation cancelled with %d block(s) open; closing them.",
            len(blocks),
        )
        raise
    except Exception as e:
        logger.warning(
            "TLDR condensation failed (%s: %s) with %d block(s) open; "
            "closing them and continuing with the uncondensed history.",
            type(e).__name__,
            e,
            len(blocks),
            exc_info=True,
        )
        raise
    finally:
        # Always close every block we opened, even if we raised or were
        # cancelled mid-iteration. Leaving a start without its matching stop
        # strands the TLDR block "in progress" on the client until the stream
        # times out.
        # put_nowait: the queue is unbounded, and awaiting here would be an
        # extra cancellation point that could skip the sentinel.
        for block in blocks.values():
            queue.put_nowait(RawContentBlockStopEvent.from_start(block))

        # Always unblock the consumer, even if we raised mid-iteration.
        queue.put_nowait(_SENTINEL)

    return result


class _OpenBlockTracker:
    """Tracks the blocks the client has actually seen, so each closes exactly once.

    Closing cannot be driven off the queue: events the consumer already took out
    of it are gone, and blocks suppressed by ``min_duration`` never reached the
    client at all. Recording what was emitted is the only view that matches what
    the client is really rendering.
    """

    def __init__(self) -> None:
        self._open: dict[str, RawContentBlockStartEvent] = {}

    def emit(self, emit_fn: Callable[[Event], None], event: Event) -> None:
        """Emit an event, then record it. Order matters.

        A start whose emission raised never reached the client, so it must not be
        tracked as open: closing it later would send a stop for a block the
        client never saw.
        """
        emit_fn(event)
        if isinstance(event, RawContentBlockStartEvent):
            self._open[event.block_id] = event
        elif isinstance(event, RawContentBlockStopEvent):
            self._open.pop(event.block_id, None)

    def close_pending(self, emit_fn: Callable[[Event], None]) -> None:
        """Close every block still open on the client.

        Pops before emitting, so a block is never stopped twice even if this runs
        more than once, and a failing ``emit_fn`` cannot wedge the loop.
        """
        if self._open:
            logger.warning(
                "Force-closing %d TLDR block(s) left open: %s. The client would "
                "otherwise render them as in-progress until the stream times out.",
                len(self._open),
                ", ".join(sorted(self._open)),
            )

        while self._open:
            block_id, start = self._open.popitem()
            try:
                emit_fn(RawContentBlockStopEvent.from_start(start))
            except Exception as e:
                # Usually the transport is already gone, which is why we got
                # here. Logged rather than raised: the remaining blocks still
                # need closing.
                logger.warning(
                    "Could not emit stop event for TLDR block %s (%s: %s).",
                    block_id,
                    type(e).__name__,
                    e,
                )


async def _consume_and_emit_with_min_duration(
    emit_fn: Callable[[Event], None],
    queue: asyncio.Queue[Event | object | None],
    min_duration: float | None = None,
    tracker: _OpenBlockTracker | None = None,
) -> None:
    tracker = tracker if tracker is not None else _OpenBlockTracker()

    if min_duration is not None:
        await asyncio.sleep(min_duration)

        buffered: list[Event | object | None] = []
        while not queue.empty():
            buffered.append(queue.get_nowait())

        has_finished = any(event is _SENTINEL for event in buffered)
        has_deltas = any(isinstance(e, RawContentBlockDeltaEvent) for e in buffered)
        if has_finished and not has_deltas:
            # Avoid emitting TLDR if the time is lower than min_duration
            return

        for event in buffered:
            if event is _SENTINEL:
                return
            # Everything the producer queues is an Event; _SENTINEL is the only
            # other value and it returned above.
            tracker.emit(emit_fn, cast(Event, event))

    while True:
        event = await queue.get()
        if event is _SENTINEL:
            break
        tracker.emit(emit_fn, cast(Event, event))


@singleton
class CondensationRequestInterceptor(ChatRequestLoopInterceptor):
    """Reduce conversation history size before iterative loop execution."""

    @inject
    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.chat.condense_strategy != "none"
        self._strategy_type = settings.chat.condense_strategy
        self._condensation_timeout = settings.chat.tldr_timeout
        self._min_duration = settings.chat.tldr_minimum_threshold_seconds

    async def intercept(self, context: ChatInterceptorContext) -> None:
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        state = context.state
        history = state.input.request.to_messages()

        if not self._enabled or not history:
            return

        token_limit = state.runtime.effective_token_limit
        if token_limit is None:
            return

        max_length = _token_limit_with_buffer(
            token_limit,
            state.input.request.condensation.token_buffer,
        )

        generator = condense_chat_history(
            **state.input.llm_kwargs.as_kwargs(),
            chat_history=history,
            tools=state.input.context_stack.all_tools(),
            strategy_type=self._strategy_type,
            max_length=max_length,
            tokenizer_fn=context.state.runtime.tokenizer_fn,
            message_to_input=context.llm.messages_to_prompt,
            condensation_timeout=self._condensation_timeout,
            model_id=state.input.request.system.model,
        )

        queue: asyncio.Queue[Event | object | None] = asyncio.Queue()
        result = _CondensationResult()
        producer = asyncio.create_task(_condensation_producer(queue, result, generator))
        tracker = _OpenBlockTracker()

        try:
            await _consume_and_emit_with_min_duration(
                emit_fn=context.emit_event,
                queue=queue,
                min_duration=self._min_duration,
                tracker=tracker,
            )
        except BaseException as e:
            if isinstance(e, asyncio.CancelledError):
                logger.debug("TLDR streaming cancelled; closing any open block.")
            else:
                logger.warning(
                    "TLDR streaming failed while emitting events (%s: %s); "
                    "closing any open block.",
                    type(e).__name__,
                    e,
                    exc_info=True,
                )
            if not producer.done():
                producer.cancel()
            with contextlib.suppress(BaseException):
                await producer
            # Nothing is draining the queue any more, so the stops the producer
            # queued will never be emitted. Close whatever the client still has
            # open, or the TLDR block hangs "in progress" until the stream times
            # out. Driven off what was emitted, so blocks the client never saw
            # are not stopped and blocks already stopped are not stopped twice.
            tracker.close_pending(context.emit_event)
            raise

        try:
            result = await producer
        except Exception as e:
            logger.error(
                "Error during condensation (%s: %s); chat will continue with the "
                "uncondensed history.",
                type(e).__name__,
                e,
                exc_info=True,
            )
            # Normally a no-op: the consumer only returns once it has drained the
            # producer's stop events. Kept so no path out of here can leave a
            # block open on the client.
            tracker.close_pending(context.emit_event)
            raise

        if result.condensed and result.chat_history is not None:
            state.input.request.messages = result.chat_history

        context.set_state(state)
