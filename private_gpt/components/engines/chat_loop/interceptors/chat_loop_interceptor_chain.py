from collections.abc import Callable, Mapping
from typing import Any, Self, cast

from pydantic import BaseModel, ConfigDict, Field

from private_gpt.components.engines.chat_loop.interceptors.chat_loop_interceptor import (
    ChatRequestLoopInterceptor,
    ChatResponseLoopInterceptor,
)

Condition = bool | Callable[[], bool]


def _evaluate(condition: Condition) -> bool:
    return condition() if callable(condition) else condition


class InterceptorChainEntry(BaseModel):
    """One named group in the chain holding N request and/or response interceptors."""

    name: str
    requests: list[ChatRequestLoopInterceptor] = Field(default_factory=list)
    responses: list[ChatResponseLoopInterceptor] = Field(default_factory=list)
    condition: Condition = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def is_active(self) -> bool:
        return _evaluate(self.condition)

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        if not deep:
            return super().model_copy(update=update, deep=deep)

        deep_copy = InterceptorChainEntry(
            name=str(self.name),
            requests=[r.model_copy(deep=True) for r in self.requests],
            responses=[r.model_copy(deep=True) for r in self.responses],
            condition=self.condition,  # shared — callable/bool are stateless
        )
        return cast(Self, deep_copy)


class ChatLoopInterceptorChain(BaseModel):
    entries: list[InterceptorChainEntry] = Field(default_factory=list)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def add(
        self,
        name: str,
        *,
        request: ChatRequestLoopInterceptor | None = None,
        response: ChatResponseLoopInterceptor | None = None,
        condition: Condition = True,
    ) -> "ChatLoopInterceptorChain":
        if request is None and response is None:
            raise ValueError(
                f"Group '{name}' must provide at least one of request or response interceptor."
            )
        self.entries.append(
            InterceptorChainEntry(
                name=name,
                requests=[request] if request else [],
                responses=[response] if response else [],
                condition=condition,
            )
        )
        return self

    def add_range(
        self,
        name: str,
        *,
        requests: list[ChatRequestLoopInterceptor] | None = None,
        responses: list[ChatResponseLoopInterceptor] | None = None,
        condition: Condition = True,
    ) -> "ChatLoopInterceptorChain":
        resolved_requests = requests or []
        resolved_responses = responses or []
        if not resolved_requests and not resolved_responses:
            raise ValueError(
                f"Group '{name}' must provide at least one request or response interceptor."
            )
        self.entries.append(
            InterceptorChainEntry(
                name=name,
                requests=resolved_requests,
                responses=resolved_responses,
                condition=condition,
            )
        )
        return self

    def clone(self) -> "ChatLoopInterceptorChain":
        return ChatLoopInterceptorChain(
            entries=[entry.model_copy(deep=True) for entry in self.entries]
        )

    @property
    def request_interceptors(self) -> list[ChatRequestLoopInterceptor]:
        return [r for e in self.entries if e.is_active for r in e.requests]

    @property
    def response_interceptors(self) -> list[ChatResponseLoopInterceptor]:
        return [r for e in self.entries if e.is_active for r in e.responses]
