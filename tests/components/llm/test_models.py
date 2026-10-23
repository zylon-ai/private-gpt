import pytest

from private_gpt.components.engines.chat.models.chat_llm_params import (
    ChatLLMParameters,
)
from private_gpt.components.llm.custom.base import (
    StructuredOutputsParams,
    normalize_structured_outputs,
)
from private_gpt.components.llm.models import (
    ReasoningEffort,
    normalize_reasoning_effort,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, ReasoningEffort.NONE),
        (ReasoningEffort.HIGH, ReasoningEffort.HIGH),
        ("high", ReasoningEffort.HIGH),
        ("HIGH", ReasoningEffort.HIGH),
    ],
)
def test_normalize_reasoning_effort(
    value: ReasoningEffort | str | None,
    expected: ReasoningEffort,
) -> None:
    assert normalize_reasoning_effort(value) is expected


def test_normalize_reasoning_effort_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="Unknown reasoning effort level"):
        normalize_reasoning_effort("unsupported")


def test_normalize_reasoning_effort_rejects_wrong_type() -> None:
    with pytest.raises(TypeError, match="must be a ReasoningEffort"):
        normalize_reasoning_effort(1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        None,
        StructuredOutputsParams(json_schema={"type": "object"}),
        {"json_schema": {"type": "object"}},
        {"json": {"type": "object"}},
        '{"json": {"type": "object"}}',
    ],
)
def test_normalize_structured_outputs(
    value: StructuredOutputsParams | dict[str, object] | str | None,
) -> None:
    normalized = normalize_structured_outputs(value)

    if value is None:
        assert normalized is None
    else:
        assert isinstance(normalized, StructuredOutputsParams)
        assert normalized.json_schema == {"type": "object"}


def test_normalize_structured_outputs_preserves_model_instance() -> None:
    value = StructuredOutputsParams(json_schema={"type": "object"})

    assert normalize_structured_outputs(value) is value


def test_chat_llm_parameters_preserves_api_shaped_structured_outputs() -> None:
    params = ChatLLMParameters.model_validate(
        {"structured_outputs": {"json": {"type": "object"}}}
    )

    assert isinstance(params.structured_outputs, StructuredOutputsParams)
    assert params.structured_outputs.json_schema == {"type": "object"}


@pytest.mark.parametrize("value", [1, "not-json", "[]"])
def test_normalize_structured_outputs_rejects_invalid_value(
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="structured_outputs"):
        normalize_structured_outputs(value)  # type: ignore[arg-type]
