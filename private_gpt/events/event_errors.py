import enum
from typing import Any

from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

from private_gpt.artifact_index.artifact_exception import InvalidFileError


class _Types(enum.StrEnum):
    INVALID_REQUEST_ERROR = "invalid_request_error"
    AUTHENTICATION_ERROR = "authentication_error"
    BILLING_ERROR = "billing_error"
    PERMISSION_ERROR = "permission_error"
    NOT_FOUND_ERROR = "not_found_error"
    REQUEST_TOO_LARGE_ERROR = "request_too_large"
    RATE_LIMIT_ERROR = "rate_limit_error"
    API_ERROR = "api_error"
    OVERLOADED_ERROR = "overloaded_error"


class _Codes(enum.StrEnum):
    INVALID_REQUEST_ERROR = "INVALID_REQUEST_ERROR"
    INVALID_REQUEST_EMPTY_MESSAGE_ERROR = "INVALID_REQUEST_EMPTY_MESSAGE_ERROR"
    INVALID_REQUEST_TOOL_SUPPORTED_ERROR = "INVALID_REQUEST_TOOL_SUPPORTED_ERROR"
    INVALID_REQUEST_REASONING_SUPPORTED_ERROR = (
        "INVALID_REQUEST_REASONING_SUPPORTED_ERROR"
    )
    INVALID_REQUEST_IMAGE_SUPPORT_ERROR = "INVALID_REQUEST_IMAGE_SUPPORT_ERROR"
    INVALID_REQUEST_IMAGE_MAX_NUM_ERROR = "INVALID_REQUEST_IMAGE_MAX_NUM_ERROR"
    INVALID_REQUEST_AUDIO_SUPPORT_ERROR = "INVALID_REQUEST_AUDIO_SUPPORT_ERROR"
    INVALID_REQUEST_AUDIO_MAX_NUM_ERROR = "INVALID_REQUEST_AUDIO_MAX_NUM_ERROR"
    INVALID_REQUEST_INVALID_MCP_ERROR = "INVALID_REQUEST_INVALID_MCP_ERROR"

    REQUEST_TOO_LARGE = "REQUEST_TOO_LARGE_ERROR"
    REQUEST_TOO_LARGE_USER_MSG = "REQUEST_TOO_LARGE_USER_MSG_ERROR"
    REQUEST_TOO_LARGE_SYSTEM_MSG = "REQUEST_TOO_LARGE_SYSTEM_MSG_ERROR"

    PERMISSION_ERROR = "PERMISSION_ERROR"
    PERMISSION_MCP_AUTH_ERROR = "PERMISSION_MCP_AUTH_ERROR"

    OVERLOADED_ERROR = "OVERLOADED_ERROR"
    OVERLOADED_CONDENSATION_ERROR = "OVERLOADED_CONDENSATION_ERROR"

    AUTHENTICATION_ERROR = "AUTHENTICATION_ERROR"
    BILLING_ERROR = "BILLING_ERROR"
    NOT_FOUND_ERROR = "NOT_FOUND_ERROR"
    RATE_LIMIT_ERROR = "RATE_LIMIT_ERROR"
    API_ERROR = "API_ERROR"


class Errors:
    Types = _Types
    Codes = _Codes

    class Base(Exception):
        error_type: _Types
        status_code: int
        default_event_code: _Codes

        def __init__(
            self,
            message: str,
            event_code: _Codes | None = None,
            status_code: int | None = None,
            original_exception: BaseException | None = None,
        ) -> None:
            super().__init__(message)
            self.event_code = (
                event_code if event_code is not None else self.default_event_code
            )
            self.status_code = (
                status_code if status_code is not None else self.status_code
            )
            self.original_exception = original_exception

        def to_dict(self) -> dict[str, Any]:
            return {
                "class": type(self).__name__,
                "message": str(self),
                "event_code": self.event_code.value,
                "status_code": self.status_code,
                "error_type": self.error_type.value,
                "original_exception": {
                    "class": type(self.original_exception).__name__,
                    "message": str(self.original_exception),
                }
                if self.original_exception is not None
                else None,
            }

        @classmethod
        def from_dict(cls, data: dict[str, Any]) -> "Errors.Base":
            error_cls = Errors._CLASS_NAME_TO_ERROR.get(data["class"], Errors.Base)
            event_code = _Codes(data["event_code"])

            original_exception: BaseException | None = None
            if raw := data.get("original_exception"):
                original_cls = Errors._CLASS_NAME_TO_EXCEPTION.get(
                    raw["class"], Exception
                )
                original_exception = original_cls(raw["message"])

            return error_cls(
                data["message"],
                event_code=event_code,
                original_exception=original_exception,
            )

    class InvalidRequest(Base):
        error_type = _Types.INVALID_REQUEST_ERROR
        status_code = 400
        default_event_code = _Codes.INVALID_REQUEST_ERROR

    class Authentication(Base):
        error_type = _Types.AUTHENTICATION_ERROR
        status_code = 401
        default_event_code = _Codes.AUTHENTICATION_ERROR

    class Billing(Base):
        error_type = _Types.BILLING_ERROR
        status_code = 402
        default_event_code = _Codes.BILLING_ERROR

    class PermissionDenied(Base):
        error_type = _Types.PERMISSION_ERROR
        status_code = 403
        default_event_code = _Codes.PERMISSION_ERROR

    class NotFound(Base):
        error_type = _Types.NOT_FOUND_ERROR
        status_code = 404
        default_event_code = _Codes.NOT_FOUND_ERROR

    class RequestTooLarge(Base):
        error_type = _Types.REQUEST_TOO_LARGE_ERROR
        status_code = 413
        default_event_code = _Codes.REQUEST_TOO_LARGE

    class RateLimit(Base):
        error_type = _Types.RATE_LIMIT_ERROR
        status_code = 429
        default_event_code = _Codes.RATE_LIMIT_ERROR

    class InternalServerError(Base):
        error_type = _Types.API_ERROR
        status_code = 500
        default_event_code = _Codes.API_ERROR

    class Overloaded(Base):
        error_type = _Types.OVERLOADED_ERROR
        status_code = 529
        default_event_code = _Codes.OVERLOADED_ERROR

    _EXCEPTION_TO_ERROR: dict[type[BaseException], type["Errors.Base"]]
    _CLASS_NAME_TO_ERROR: dict[str, type["Errors.Base"]]
    _CLASS_NAME_TO_EXCEPTION: dict[str, type[Exception]]

    @staticmethod
    def build(
        exception: BaseException,
        event_code: _Codes | None = None,
    ) -> "Errors.Base":
        if isinstance(exception, Errors.Base):
            return exception
        error_cls = Errors._EXCEPTION_TO_ERROR.get(
            type(exception), Errors.InternalServerError
        )
        return error_cls(
            str(exception),
            event_code=event_code,
            original_exception=exception,
        )


Errors._EXCEPTION_TO_ERROR = {
    ValueError: Errors.InvalidRequest,
    ValidationError: Errors.InvalidRequest,
    RequestValidationError: Errors.InvalidRequest,
    ImportError: Errors.InvalidRequest,
    ModuleNotFoundError: Errors.InvalidRequest,
    PermissionError: Errors.PermissionDenied,
    InvalidFileError: Errors.InvalidRequest,
    Errors.NotFound: Errors.NotFound,
    Errors.RequestTooLarge: Errors.RequestTooLarge,
    Errors.RateLimit: Errors.RateLimit,
    Errors.Authentication: Errors.Authentication,
    Errors.Billing: Errors.Billing,
    Errors.PermissionDenied: Errors.PermissionDenied,
    TypeError: Errors.RequestTooLarge,
    KeyError: Errors.RequestTooLarge,
    ConnectionError: Errors.PermissionDenied,
    Errors.Overloaded: Errors.Overloaded,
    MemoryError: Errors.Overloaded,
}

Errors._CLASS_NAME_TO_ERROR = {
    cls.__name__: cls for cls in Errors._EXCEPTION_TO_ERROR.values()
}

Errors._CLASS_NAME_TO_EXCEPTION = {
    cls.__name__: cls
    for cls in [
        ValueError,
        ImportError,
        ModuleNotFoundError,
        PermissionError,
        TypeError,
        KeyError,
        ConnectionError,
        MemoryError,
        RuntimeError,
        Exception,
    ]
} | {cls.__name__: cls for cls in Errors._EXCEPTION_TO_ERROR.values()}
