from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel

from private_gpt.components.engines.chat_loop.models.chat_loop_interceptor_context import (
    ChatLoopInterceptorContext,
)
from private_gpt.events.models import Event


class ChatRequestLoopInterceptor(ABC, BaseModel):
    """Transform loop context before inference starts."""

    @abstractmethod
    async def intercept(self, context: ChatLoopInterceptorContext) -> None:
        """Mutate context state and emit events when needed."""

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        # By default, return a shallow copy
        # since interceptors are stateless and can be shared.
        return self


class ChatResponseLoopInterceptor(ABC, BaseModel):
    """Process events emitted during inference, and mutate loop context as needed."""

    async def on_iteration_start(self, context: ChatLoopInterceptorContext) -> None:
        """Called once before each iteration — reset per-iteration state here."""
        return

    async def on_iteration_end(self, context: ChatLoopInterceptorContext) -> None:
        """Called once after each iteration."""
        return

    @abstractmethod
    async def intercept_event(
        self,
        event: Event,
        context: ChatLoopInterceptorContext,
    ) -> Event | None:
        """Process one event. Mutate context.state as needed."""

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        # By default, return a shallow copy
        # since interceptors are stateless and can be shared.
        return self
