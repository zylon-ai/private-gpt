from collections.abc import Callable
from typing import Any

from llama_index.core.llms.function_calling import FunctionCallingLLM
from pydantic import BaseModel, ConfigDict, Field

from private_gpt.components.engines.chat.models.chat_phase import (
    InterceptorPhase,
)
from private_gpt.components.engines.chat.models.chat_state import (
    ChatState,
)
from private_gpt.events.models import Event


class ChatInterceptorContext(BaseModel):
    """Carry state, llm runtime, and helpers across interceptors."""

    state: ChatState
    llm: FunctionCallingLLM
    phase: InterceptorPhase
    emit_fn: Callable[[Event], None]
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def emit_event(self, event: Event) -> None:
        """Emit one event immediately to the loop stream."""
        self.emit_fn(event)

    def set_state(self, state: ChatState) -> None:
        """Replace context state after interceptor transformation."""
        self.state = state
