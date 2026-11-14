"""Regression coverage for error categories and cross-process reconstruction."""

import pytest

from private_gpt.components.skills.errors import (
    SkillDomainError,
    SkillErrorCode,
    SkillValidationErrors,
)
from private_gpt.events.event_errors import Errors


@pytest.mark.parametrize("code", list(SkillErrorCode))
def test_skill_domain_error_category(code: SkillErrorCode) -> None:
    original = SkillDomainError(code, "Correct the skill bundle")

    error = Errors.build(original)

    expected = (
        Errors.RequestTooLarge
        if code == SkillErrorCode.BUNDLE_TOO_LARGE
        else Errors.InvalidRequest
    )
    assert type(error) is expected
    assert str(error) == str(original)
    assert error.original_exception is original


@pytest.mark.parametrize("oversized", [False, True])
def test_aggregated_skill_error_category(oversized: bool) -> None:
    errors = [SkillDomainError(SkillErrorCode.NAME_REQUIRED, "Add a name")]
    if oversized:
        errors.append(
            SkillDomainError(SkillErrorCode.BUNDLE_TOO_LARGE, "Reduce bundle size")
        )
    original = SkillValidationErrors(errors)

    error = Errors.build(original)

    assert type(error) is (
        Errors.RequestTooLarge if oversized else Errors.InvalidRequest
    )
    assert str(error) == str(original)
    assert error.original_exception is original


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        (TypeError("unexpected type"), Errors.InternalServerError),
        (KeyError("missing key"), Errors.InternalServerError),
        (RuntimeError("internal failure"), Errors.InternalServerError),
        (ConnectionError("connection lost"), Errors.Overloaded),
        (TimeoutError("request timed out"), Errors.Overloaded),
    ],
)
def test_error_category_and_round_trip(
    original: Exception, expected: type[Errors.Base]
) -> None:
    error = Errors.build(original)
    restored = Errors.Base.from_dict(error.to_dict())

    assert type(error) is expected
    assert error.original_exception is original
    assert type(restored) is expected
    assert restored.status_code == expected.status_code
    assert restored.event_code == expected.default_event_code
    assert type(restored.original_exception) is type(original)
    assert restored.original_exception.args == (str(original),)


def test_explicit_error_and_event_code_are_preserved() -> None:
    original = Errors.InvalidRequest(
        "Provide a message",
        event_code=Errors.Codes.INVALID_REQUEST_EMPTY_MESSAGE_ERROR,
    )

    assert Errors.build(original) is original
    mapped = Errors.build(
        SkillDomainError(SkillErrorCode.NAME_REQUIRED, "Add a name"),
        event_code=Errors.Codes.INVALID_REQUEST_EMPTY_MESSAGE_ERROR,
    )
    assert mapped.event_code == Errors.Codes.INVALID_REQUEST_EMPTY_MESSAGE_ERROR
