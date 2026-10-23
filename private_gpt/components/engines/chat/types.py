from collections.abc import Awaitable, Callable, Sequence
from typing import TypeAlias

from private_gpt.components.chat.models.chat_config_models import ToolSpec

SystemPromptFn: TypeAlias = (
    Callable[..., str | None] | Callable[..., Awaitable[str | None]]
)
ToolsFn: TypeAlias = (
    Callable[..., Sequence[ToolSpec]] | Callable[..., Awaitable[Sequence[ToolSpec]]]
)
