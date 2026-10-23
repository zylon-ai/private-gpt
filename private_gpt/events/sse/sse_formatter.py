from collections.abc import Iterator

from private_gpt.events.models import Event


class SSEFormatter:
    @staticmethod
    def format_event(
        event_type: str | None, data: str, include_wrap_event: bool = True
    ) -> str:
        event_line = f"event: {event_type}\n" if event_type else ""
        data_line = f"data: {data}"
        if include_wrap_event:
            data_line += "\n\n"
        return event_line + data_line

    @classmethod
    def to_sse_event(cls, event: Event) -> str:
        return cls.format_event(
            event_type=event.type,
            data=(
                event.model_dump_json()
                if hasattr(event, "model_dump_json")
                else str(event)
            ),
        )

    @classmethod
    def to_sse_stream(cls, sse_events: Iterator[Event]) -> Iterator[str]:
        for event in sse_events:
            yield cls.to_sse_event(event)
