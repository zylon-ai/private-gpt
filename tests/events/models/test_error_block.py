"""Client errors carry recovery text; server errors retain their category title."""

import pytest
from fastapi.exceptions import RequestValidationError

from private_gpt.events.event_errors import Errors
from private_gpt.events.models import ErrorBlock


@pytest.mark.parametrize(
    "error_cls",
    [
        Errors.InvalidRequest,
        Errors.Authentication,
        Errors.Billing,
        Errors.PermissionDenied,
        Errors.NotFound,
        Errors.RequestTooLarge,
        Errors.RateLimit,
    ],
)
def test_client_error_message_is_explanation(error_cls: type[Errors.Base]) -> None:
    block = ErrorBlock.from_exception(error_cls("Correct the request and retry"))

    assert block.message == "Correct the request and retry"
    assert block.detail is not None
    assert block.detail.explanation == block.message
    assert block.detail.code == error_cls.default_event_code


@pytest.mark.parametrize("error_cls", [Errors.InternalServerError, Errors.Overloaded])
def test_server_error_message_remains_title(error_cls: type[Errors.Base]) -> None:
    block = ErrorBlock.from_exception(error_cls("internal diagnostic text"))

    assert block.message == error_cls.error_type.replace("_", " ").title()
    assert block.detail is not None
    assert block.detail.explanation == "internal diagnostic text"


def test_structured_validation_explanation_keeps_title() -> None:
    details = [{"type": "missing", "loc": ("body", "name"), "msg": "Field required"}]
    block = ErrorBlock.from_exception(
        Errors.InvalidRequest(
            "Request validation failed",
            original_exception=RequestValidationError(details),
        )
    )

    assert block.message == "Invalid Request Error"
    assert block.detail is not None
    assert block.detail.explanation == [
        {"type": "missing", "loc": "('body', 'name')", "msg": "Field required"}
    ]
