from collections.abc import AsyncGenerator
from typing import Literal

from private_gpt.events.interceptors.base_event_interceptor import BaseEventInterceptor
from private_gpt.events.models import (
    Event,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
)


class FilterZylonEventInterceptor(BaseEventInterceptor):
    _response_mode: Literal["zylon", "anthropic"]

    def __init__(self, response_mode: Literal["zylon", "anthropic"]) -> None:
        self._response_mode = response_mode

    async def intercept(
        self, gen: AsyncGenerator[Event, None]
    ) -> AsyncGenerator[Event, None]:
        """Filters out Zylon-specific events from the event stream.

        Args:
            gen: An async generator yielding Event objects.

        Yields:
            Event: The filtered Event objects.
        """

        async def coro() -> AsyncGenerator[Event, None]:
            active_blocks: set[str] = set()
            async for event in gen:
                new_event = event.prune_content_block_by_response_mode(
                    self._response_mode
                )
                if not new_event:
                    continue

                if isinstance(event, RawContentBlockStartEvent):
                    active_blocks.add(event.block_id)
                    yield event
                elif isinstance(event, RawContentBlockStopEvent):
                    if event.block_id in active_blocks:
                        active_blocks.remove(event.block_id)
                        yield event
                    else:
                        continue  # Edge case: If the content block never started, skip the end event
                else:
                    yield event

        return coro()
