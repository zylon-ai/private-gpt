# reasoning_budget.py

from collections.abc import AsyncGenerator, Callable, Coroutine, Sequence
from functools import wraps
from typing import Any

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
)
from llama_index.core.tools import BaseTool

from private_gpt.components.llm.models import ReasoningEffort
from private_gpt.events.models import StopReasonEnum

_REASONING_BUDGET_KWARG = "reasoning_budget"
_USAGE_KEYS = ("input_tokens", "output_tokens")

_EFFORT_BUDGET_RATIO: dict[ReasoningEffort, float] = {
    ReasoningEffort.LOW: 0.8,
    ReasoningEffort.MEDIUM: 0.8,
    ReasoningEffort.HIGH: 0.8,
    ReasoningEffort.MAX: 0.8,
}


def _resolve_stop_reason(response: ChatResponse) -> str | None:
    return response.additional_kwargs.get(
        "stop_reason"
    ) or response.message.additional_kwargs.get("stop_reason")


def _is_truncated_thinking(response: ChatResponse) -> bool:
    stop = _resolve_stop_reason(response)
    no_content = not (response.message.content or "").strip()
    no_tool_calls = not response.message.additional_kwargs.get("tool_calls")
    return stop == StopReasonEnum.MAX_TOKENS and no_content and no_tool_calls


def _resolve_usage(response: ChatResponse, key: str) -> int:
    return (
        response.additional_kwargs.get(key)
        or response.message.additional_kwargs.get(key)
        or 0
    )


def _merge_responses(phase1: ChatResponse, phase2: ChatResponse) -> ChatResponse:
    merged_message_kwargs = dict(phase2.message.additional_kwargs)
    if thinking := phase1.message.additional_kwargs.get("thinking"):
        merged_message_kwargs["thinking"] = thinking

    merged_kwargs = dict(phase2.additional_kwargs)
    for key in _USAGE_KEYS:
        merged_kwargs[key] = _resolve_usage(phase1, key) + _resolve_usage(phase2, key)

    return ChatResponse(
        message=ChatMessage(
            role=phase2.message.role,
            content=phase2.message.content,
            additional_kwargs=merged_message_kwargs,
        ),
        additional_kwargs=merged_kwargs,
    )


def _extract_budget(**kwargs: Any) -> tuple[int | None, dict[str, Any]]:
    budget = kwargs.get(_REASONING_BUDGET_KWARG)
    if budget is not None and not isinstance(budget, (int, float)):
        raise ValueError(
            f"Invalid reasoning_effort budget: {budget}. Must be a number (int or float)."
        )

    clean = {k: v for k, v in kwargs.items() if k != _REASONING_BUDGET_KWARG}
    return (int(budget) if budget is not None else None), clean


def _build_continue_messages(
    messages: Sequence[ChatMessage],
    phase1: ChatResponse,
) -> list[ChatMessage]:
    return [*messages, phase1.message]


def _effective_reasoning(
    reasoning_effort: ReasoningEffort,
    budget: int | None,
) -> ReasoningEffort:
    return ReasoningEffort.NONE if budget == 0 else reasoning_effort


def _get_max_tokens(self: Any, **kwargs: Any) -> int:
    specific_max_tokens = kwargs.get("max_tokens")
    if specific_max_tokens is not None:
        return int(specific_max_tokens)

    max_tokens: int = (
        self.get_metadata(**kwargs).num_output
        if hasattr(self, "get_metadata")
        else self.metadata.num_output
    )
    return max_tokens


def _resolve_budget(
    self: Any,
    explicit: int | None,
    reasoning_effort: ReasoningEffort,
    **kwargs: dict[str, Any],
) -> int | None:
    if explicit is not None:
        return explicit
    ratio = _EFFORT_BUDGET_RATIO.get(reasoning_effort)
    if ratio is None:
        return None
    total = _get_max_tokens(self, reasoning_effort=reasoning_effort, **kwargs)
    if not total:
        return None
    return max(1, int(total * ratio))


def _phase2_kwargs(max_tokens: int, budget: int, **kwargs: Any) -> Any:
    if not max_tokens:
        return kwargs

    kwargs.pop("max_tokens", None)
    return {**kwargs, "max_tokens": max(1, max_tokens - budget)}


def with_reasoning_budget_chat(
    fn: Callable[..., ChatResponse]
) -> Callable[..., ChatResponse]:
    """Decorator for sync `chat`."""

    @wraps(fn)
    def wrapper(
        self: Any,
        messages: Sequence[ChatMessage],
        tools: Sequence[BaseTool] | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
        **kwargs: Any,
    ) -> ChatResponse:
        # Check if reasoning_effort budget is enabled for this model
        if not getattr(self, "allow_reasoning_budget", False):
            return fn(self, messages, tools=tools, reasoning_effort=reasoning_effort, **kwargs)  # type: ignore[return-value]

        explicit, kwargs = _extract_budget(**kwargs)
        budget = _resolve_budget(self, explicit, reasoning_effort, **kwargs)
        reasoning_effort = _effective_reasoning(reasoning_effort, budget)
        current_max_tokens = _get_max_tokens(
            self, reasoning_effort=reasoning_effort, **kwargs
        )

        if reasoning_effort == ReasoningEffort.NONE or not budget:
            return fn(self, messages, tools=tools, reasoning_effort=reasoning_effort, **kwargs)  # type: ignore[return-value]

        kwargs.pop("max_tokens", None)
        phase1: ChatResponse = fn(
            self,
            messages,
            tools=tools,
            reasoning_effort=reasoning_effort,
            max_tokens=budget,
            **kwargs,
        )
        if not _is_truncated_thinking(phase1):
            return phase1

        phase2: ChatResponse = fn(
            self,
            _build_continue_messages(messages, phase1),
            tools=tools,
            reasoning_effort=ReasoningEffort.NONE,
            **_phase2_kwargs(current_max_tokens, budget, **kwargs),
        )
        return _merge_responses(phase1, phase2)

    return wrapper  # type: ignore[return-value]


def with_reasoning_budget_stream_chat(
    fn: Callable[..., ChatResponseGen]
) -> Callable[..., ChatResponseGen]:
    """Decorator for sync `stream_chat`, which returns a generator."""

    @wraps(fn)
    def wrapper(
        self: Any,
        messages: Sequence[ChatMessage],
        tools: Sequence[BaseTool] | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
        **kwargs: Any,
    ) -> ChatResponseGen:
        # Check if reasoning_effort budget is enabled for this model
        if not getattr(self, "allow_reasoning_budget", False):
            return fn(self, messages, tools=tools, reasoning_effort=reasoning_effort, **kwargs)  # type: ignore[return-value]

        explicit, kwargs = _extract_budget(**kwargs)
        budget = _resolve_budget(self, explicit, reasoning_effort, **kwargs)
        reasoning_effort = _effective_reasoning(reasoning_effort, budget)
        current_max_tokens = _get_max_tokens(
            self, reasoning_effort=reasoning_effort, **kwargs
        )

        if reasoning_effort == ReasoningEffort.NONE or not budget:
            return fn(self, messages, tools=tools, reasoning_effort=reasoning_effort, **kwargs)  # type: ignore[return-value]

        def _gen() -> ChatResponseGen:
            kwargs.pop("max_tokens", None)
            phase1_last: ChatResponse | None = None

            for item in fn(
                self,
                messages,
                tools=tools,
                reasoning_effort=reasoning_effort,
                max_tokens=budget,
                **kwargs,
            ):
                phase1_last = item
                yield item

            if phase1_last is None or not _is_truncated_thinking(phase1_last):
                return

            prev: ChatResponse | None = None
            for item in fn(
                self,
                _build_continue_messages(messages, phase1_last),
                tools=tools,
                reasoning_effort=ReasoningEffort.NONE,
                **_phase2_kwargs(current_max_tokens, budget, **kwargs),
            ):
                if prev is not None:
                    yield prev
                prev = item

            if prev is not None:
                yield _merge_responses(phase1_last, prev)

        return _gen()

    return wrapper  # type: ignore[return-value]


_AChatFn = Callable[..., Coroutine[Any, Any, ChatResponse]]


def with_reasoning_budget_achat(fn: _AChatFn) -> _AChatFn:
    """Decorator for async `achat`."""

    @wraps(fn)
    async def wrapper(
        self: Any,
        messages: Sequence[ChatMessage],
        tools: Sequence[BaseTool] | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
        **kwargs: Any,
    ) -> ChatResponse:
        # Check if reasoning_effort budget is enabled for this model
        if not getattr(self, "allow_reasoning_budget", False):
            return await fn(
                self, messages, tools=tools, reasoning_effort=reasoning_effort, **kwargs
            )

        explicit, kwargs = _extract_budget(**kwargs)
        budget = _resolve_budget(self, explicit, reasoning_effort, **kwargs)
        reasoning_effort = _effective_reasoning(reasoning_effort, budget)
        current_max_tokens = _get_max_tokens(
            self, reasoning_effort=reasoning_effort, **kwargs
        )

        if reasoning_effort == ReasoningEffort.NONE or not budget:
            return await fn(
                self, messages, tools=tools, reasoning_effort=reasoning_effort, **kwargs
            )

        kwargs.pop("max_tokens", None)
        phase1: ChatResponse = await fn(
            self,
            messages,
            tools=tools,
            reasoning_effort=reasoning_effort,
            max_tokens=budget,
            **kwargs,
        )
        if not _is_truncated_thinking(phase1):
            return phase1

        phase2: ChatResponse = await fn(
            self,
            _build_continue_messages(messages, phase1),
            tools=tools,
            reasoning_effort=ReasoningEffort.NONE,
            **_phase2_kwargs(current_max_tokens, budget, **kwargs),
        )
        return _merge_responses(phase1, phase2)

    return wrapper  # type: ignore[return-value]


_AStreamChatFn = Callable[..., Coroutine[Any, Any, AsyncGenerator[ChatResponse, None]]]


def with_reasoning_budget_astream_chat(fn: _AStreamChatFn) -> _AStreamChatFn:
    """Decorator for async `astream_chat`."""

    @wraps(fn)
    async def wrapper(
        self: Any,
        messages: Sequence[ChatMessage],
        tools: Sequence[BaseTool] | None = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatResponse, None]:
        # Check if reasoning_effort budget is enabled for this model
        if not getattr(self, "allow_reasoning_budget", False):
            return await fn(  # type: ignore[return-value]
                self, messages, tools=tools, reasoning_effort=reasoning_effort, **kwargs
            )

        explicit, kwargs = _extract_budget(**kwargs)
        budget = _resolve_budget(self, explicit, reasoning_effort, **kwargs)
        reasoning_effort = _effective_reasoning(reasoning_effort, budget)
        current_max_tokens = _get_max_tokens(
            self, reasoning_effort=reasoning_effort, **kwargs
        )

        if reasoning_effort == ReasoningEffort.NONE or not budget:
            return await fn(  # type: ignore[return-value]
                self, messages, tools=tools, reasoning_effort=reasoning_effort, **kwargs
            )

        async def _gen() -> AsyncGenerator[ChatResponse, None]:
            kwargs.pop("max_tokens", None)
            phase1_last: ChatResponse | None = None

            async for item in await fn(
                self,
                messages,
                tools=tools,
                reasoning_effort=reasoning_effort,
                max_tokens=budget,
                **kwargs,
            ):
                phase1_last = item
                yield item

            if phase1_last is None or not _is_truncated_thinking(phase1_last):
                return

            prev: ChatResponse | None = None
            async for item in await fn(
                self,
                _build_continue_messages(messages, phase1_last),
                tools=tools,
                reasoning_effort=ReasoningEffort.NONE,
                **_phase2_kwargs(current_max_tokens, budget, **kwargs),
            ):
                if prev is not None:
                    yield prev
                prev = item

            if prev is not None:
                yield _merge_responses(phase1_last, prev)

        return _gen()  # type: ignore[return-value]

    return wrapper  # type: ignore[return-value]
