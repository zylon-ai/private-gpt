from enum import StrEnum


class InterceptorPhase(StrEnum):
    """Represent interceptor execution phase in the loop."""

    VALIDATION = "validation"
    BEFORE_ITERATION = "before_iteration"
    BEFORE_TOOL = "before_tool"
    STREAMING = "streaming"
    AFTER_TOOL = "after_tool"
    AFTER_ITERATION = "after_iteration"


class TimelinePhase(StrEnum):
    """Represent timeline checkpoints captured by the loop state."""

    START = "start"
    BEFORE_INTERCEPTORS = "before_interceptors"
    AFTER_INTERCEPTORS = "after_interceptors"
    BEFORE_LLM = "before_llm"
    AFTER_LLM = "after_llm"
    AFTER_TOOLS = "after_tools"
    STOP = "stop"
