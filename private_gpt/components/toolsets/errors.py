"""Define typed errors for the toolsets module."""

from enum import StrEnum


class ToolsetsErrorCode(StrEnum):
    """Define stable error codes for toolset module failures."""

    INVALID_TOOLSET = "INVALID_TOOLSET"


class ToolsetsDomainError(Exception):
    """Represent a typed toolsets domain error with stable code and message."""

    def __init__(self, code: ToolsetsErrorCode, message: str) -> None:
        """Initialize the typed toolsets domain error."""
        self.code = code
        self.message = message
        super().__init__(message)


class InvalidToolSetError(ToolsetsDomainError):
    """Raise when a toolset is invalid or has duplicate tool names."""

    def __init__(self, message: str) -> None:
        """Initialize an invalid-toolset error."""
        super().__init__(ToolsetsErrorCode.INVALID_TOOLSET, message)
