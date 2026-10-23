import enum


class ReasoningEffort(enum.StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"
    XHIGH = "xhigh"

    @classmethod
    def from_str(cls, effort_str: str) -> "ReasoningEffort":
        effort_str = effort_str.lower()
        if effort_str in cls._value2member_map_:
            return cls(effort_str)
        raise ValueError(f"Unknown reasoning effort level: {effort_str}")

    @property
    def is_thinking_enabled(self) -> bool:
        return self != ReasoningEffort.NONE


def normalize_reasoning_effort(
    reasoning_effort: ReasoningEffort | str | None,
) -> ReasoningEffort:
    """Normalize values crossing untyped or serialized LLM boundaries."""
    if reasoning_effort is None:
        return ReasoningEffort.NONE
    if isinstance(reasoning_effort, ReasoningEffort):
        return reasoning_effort
    if isinstance(reasoning_effort, str):
        return ReasoningEffort.from_str(reasoning_effort)
    raise TypeError(
        "reasoning_effort must be a ReasoningEffort, string, or None; "
        f"got {type(reasoning_effort).__name__}"
    )


def _get_exception_types() -> tuple[type[BaseException], ...]:
    base_exceptions = (ConnectionError, TimeoutError, OSError)

    try:
        from grpc.aio import AioRpcError  # ty:ignore[unresolved-import]
        from tritonclient.utils import (  # ty:ignore[unresolved-import]
            InferenceServerException,
        )

        return *base_exceptions, AioRpcError, InferenceServerException
    except ImportError:
        return base_exceptions


MODEL_NOT_AVAILABLE_EXCEPTION_TYPES = _get_exception_types()
