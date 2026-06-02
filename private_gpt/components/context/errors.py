"""Define typed errors for context assembly validation."""

from enum import StrEnum


class ContextErrorCode(StrEnum):
    """Define stable error codes for context module failures."""

    TOOL_NAME_CONFLICT = "TOOL_NAME_CONFLICT"


class ContextDomainError(Exception):
    """Represent a typed context domain error with stable code and message."""

    def __init__(self, code: ContextErrorCode, message: str) -> None:
        """Initialize the typed context domain error."""
        self.code = code
        self.message = message
        super().__init__(message)


class ToolNameConflictError(ContextDomainError):
    """Raise when duplicate tool names are detected across tool layers."""

    def __init__(self, message: str) -> None:
        """Initialize a tool-name-conflict error."""
        super().__init__(ContextErrorCode.TOOL_NAME_CONFLICT, message)
