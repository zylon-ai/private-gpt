import builtins
import json
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from private_gpt.events.event_errors import Errors
from private_gpt.events.models._base import BaseContentBlock, StandardContentProtocol


class ErrorDetail(BaseModel):
    code: str | None = Field(default=None, description="Detailed error code")
    explanation: str | list[dict[str, object]] | None = Field(
        default=None, description="Explanation text or structured validation output"
    )


def _normalize_validation_errors(value: Any) -> list[dict[str, object]] | None:
    if not isinstance(value, list):
        return None

    normalized: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            normalized.append({"msg": str(item)})
            continue

        output: dict[str, object] = {}
        for k, v in item.items():
            if isinstance(v, str | int | float | bool) or v is None:
                output[str(k)] = v
            elif isinstance(v, list):
                output[str(k)] = [str(x) for x in v]
            elif isinstance(v, dict):
                output[str(k)] = {str(sk): str(sv) for sk, sv in v.items()}
            else:
                output[str(k)] = str(v)
        normalized.append(output)
    return normalized


class ErrorBlock(BaseContentBlock, StandardContentProtocol):
    """Error payload for SSE error events."""

    type: str = Field(description="Error type identifier")
    message: str = Field(description="Human-readable error description")
    detail: ErrorDetail | None = Field(
        default=None,
        description=(
            "Structured error detail payload. For 422 validation errors, this keeps "
            "the original validation output."
        ),
    )

    @classmethod
    def from_exception(cls, error: BaseException) -> "ErrorBlock":
        wrapped_exception = Errors.build(error)
        if wrapped_exception.original_exception is not None:
            original = wrapped_exception.original_exception
            errors_method = getattr(original, "errors", None)
            if callable(errors_method):
                try:
                    parsed = _normalize_validation_errors(errors_method())
                    explanation = parsed if parsed is not None else str(original)
                except Exception:
                    explanation = str(original)
            else:
                explanation = str(original)
        else:
            explanation = str(error)

        return cls(
            type=wrapped_exception.error_type,
            message=wrapped_exception.error_type.replace("_", " ").title(),
            detail=ErrorDetail(
                code=wrapped_exception.event_code,
                explanation=explanation,
            )
            if wrapped_exception.event_code or explanation is not None
            else None,
        )

    @classmethod
    def from_defaults(cls) -> "ErrorBlock":
        return cls.from_exception(Exception())


class FatalError(BaseModel, StandardContentProtocol):
    """Top-level fatal error response."""

    type: Literal["error"] = Field(default="error")
    error: ErrorBlock
    request_id: str | None = Field(default=None)
    exception: BaseException | None = Field(default=None)

    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True

    @model_validator(mode="before")
    @classmethod
    def _deserialize_exception(cls, values: Any) -> Any:
        if not isinstance(values, dict) or "exception" not in values:
            return values
        exc_data = values.pop("exception")
        if isinstance(exc_data, Exception):
            values["exception"] = exc_data
        elif isinstance(exc_data, dict):
            exc_type = exc_data.get("type", "")
            exc_msg = exc_data.get("message", "")
            values["exception"] = cls._get_exception_class(exc_type)(exc_msg)
        return values

    @classmethod
    def _get_exception_class(cls, name: str) -> builtins.type[BaseException]:
        registry: dict[str, builtins.type[BaseException]] = {
            "ValueError": ValueError,
            "ImportError": ImportError,
            "ModuleNotFoundError": ModuleNotFoundError,
            "RuntimeError": RuntimeError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
            "OSError": OSError,
            "IOError": IOError,
            "SystemError": SystemError,
            "Exception": Exception,
            "MemoryError": MemoryError,
            "BaseException": BaseException,
            "FileNotFoundError": FileNotFoundError,
            "PermissionError": PermissionError,
            "ConnectionError": ConnectionError,
            "BrokenPipeError": BrokenPipeError,
            "TimeoutError": TimeoutError,
        }
        custom_errors = set(Errors._EXCEPTION_TO_ERROR.values())
        for error_cls in custom_errors:
            registry[error_cls.__name__] = error_cls
            registry[f"{error_cls.__name__}Error"] = error_cls
        registry[Errors.InternalServerError.__name__] = Errors.InternalServerError
        return registry.get(name, RuntimeError)

    @classmethod
    def from_exception(
        cls, error: BaseException, request_id: str | None = None
    ) -> "FatalError":
        return cls(
            error=ErrorBlock.from_exception(error),
            exception=error,
            request_id=request_id,
        )

    @classmethod
    def from_defaults(cls) -> "FatalError":
        return cls(error=ErrorBlock.from_defaults())

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        exclude = set(kwargs.pop("exclude", None) or set())
        exclude.add("exception")
        data = super().model_dump(exclude=exclude, **kwargs)
        if self.exception:
            data["exception"] = {
                "type": type(self.exception).__name__,
                "message": str(self.exception),
            }
        return data

    def model_dump_json(self, **kwargs: Any) -> str:
        return json.dumps(self.model_dump(**kwargs), ensure_ascii=False)

    def prune_content_block_by_response_mode(
        self, response_mode: Literal["anthropic", "zylon"]
    ) -> Self | None:
        return self
