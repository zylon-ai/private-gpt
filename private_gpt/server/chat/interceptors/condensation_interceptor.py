import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Callable
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

        for block in blocks.values():
            await queue.put(RawContentBlockStopEvent.from_start(block))

    finally:
        # Always unblock the consumer, even if we raised mid-iteration.
        await queue.put(_SENTINEL)

    return result


async def _consume_and_emit_with_min_duration(
    emit_fn: Callable[[Event], None],
    queue: asyncio.Queue[Event | object | None],
    min_duration: float | None = None,
) -> None:
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
            emit_fn(event)  # type: ignore

    while True:
        event = await queue.get()
        if event is _SENTINEL:
            break
        emit_fn(event)  # type: ignore


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

        generator = condense_chat_history(
            **state.input.llm_kwargs,
            chat_history=history,
            tools=state.input.context_stack.all_tools(),
            strategy_type=self._strategy_type,
            max_length=context.state.runtime.effective_token_limit,
            tokenizer_fn=context.state.runtime.tokenizer_fn,
            message_to_input=context.llm.messages_to_prompt,
            condensation_timeout=self._condensation_timeout,
            model_id=state.input.request.system.model,
        )

        queue: asyncio.Queue[Event | object | None] = asyncio.Queue()
        result = _CondensationResult()
        producer = asyncio.create_task(_condensation_producer(queue, result, generator))

        try:
            await _consume_and_emit_with_min_duration(
                emit_fn=context.emit_event,
                queue=queue,
                min_duration=self._min_duration,
            )
        except BaseException:
            if not producer.done():
                producer.cancel()
            with contextlib.suppress(Exception):
                await producer
            raise

        try:
            result = await producer
        except Exception as e:
            logger.error(f"Error during condensation: {e}")
            raise

        if result.condensed and result.chat_history is not None:
            state.input.request.messages = result.chat_history

        context.set_state(state)
