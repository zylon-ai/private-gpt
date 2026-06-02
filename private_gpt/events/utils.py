from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from private_gpt.events.models import (
    ContentBlockType,
    Event,
    FatalError,
    Message,
    StopReasonEnum,
    Usage,
)
from private_gpt.events.sse.sse_formatter import SSEFormatter


def to_message(
    content: list[ContentBlockType] | None,
    exception: BaseException | None = None,
    stop_reason: str | StopReasonEnum | None = None,
    usage: Usage | dict[str, Any] | None = None,
) -> Message | FatalError:
    if isinstance(exception, BaseException):
        return FatalError.from_exception(exception)

    return Message(
        content=content or [],
        stop_reason=str(
            StopReasonEnum(stop_reason)
            if stop_reason and isinstance(stop_reason, str)
            else (stop_reason or StopReasonEnum.END_TURN)
        ),
        usage=usage
        if isinstance(usage, Usage)
        else Usage(**usage)
        if usage
        else Usage(),
    )


async def to_sse_stream(
    event_generator: AsyncGenerator[Event, None],
) -> AsyncIterator[str]:
    async for event in event_generator:
        yield SSEFormatter.to_sse_event(event)
