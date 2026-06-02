from injector import inject, singleton

from private_gpt.components.streaming.providers.stream_service import StreamService
from private_gpt.components.streaming.registry import _PROVIDERS, register_stream
from private_gpt.settings.settings import Settings

__all__ = ["StreamComponent", "register_stream"]


@singleton
class StreamComponent:

    stream: StreamService

    @inject
    def __init__(self, settings: Settings) -> None:
        provider = _PROVIDERS.get(settings.stream.broker)
        if provider is None:
            raise ValueError(
                f"Unsupported streaming provider: {settings.stream.broker}"
            )
        self.stream = provider(settings)
