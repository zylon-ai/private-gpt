import pytest

from private_gpt.utils.token import calculate_maximum_token_expansion

# `context_window` is always a real int: llama_index's LLMMetadata rejects None.
MODEL_WINDOW = 8192


@pytest.mark.parametrize(
    ("token_limit", "maximum_context_length", "expected"),
    [
        # No explicit limit: only the model window bounds the budget.
        (None, None, MODEL_WINDOW),
        (None, 2048, 2048),
        (None, 32_768, MODEL_WINDOW),
        # A caller-supplied limit is a cap, not a target: the tighter limit wins.
        (2048, 32_768, 2048),
        (2048, None, 2048),
        # 0 keeps its "no limit configured" meaning, as in chat settings.
        (0, 32_768, 0),
    ],
)
def test_token_budget_is_the_tightest_limit(
    token_limit: int | None,
    maximum_context_length: int | None,
    expected: int,
) -> None:
    budget = calculate_maximum_token_expansion(
        token_limit=token_limit,
        context_window=MODEL_WINDOW,
        maximum_context_length=maximum_context_length,
    )

    assert budget == expected


@pytest.mark.parametrize(
    ("token_limit", "maximum_context_length"),
    [
        (None, None),
        (2048, 32_768),
        (32_768, 2048),
        (2048, None),
        (None, 2048),
        (32_768, 32_768),
    ],
)
def test_token_budget_never_exceeds_a_limit(
    token_limit: int | None,
    maximum_context_length: int | None,
) -> None:
    budget = calculate_maximum_token_expansion(
        token_limit=token_limit,
        context_window=MODEL_WINDOW,
        maximum_context_length=maximum_context_length,
    )

    assert budget <= MODEL_WINDOW
    if token_limit is not None:
        assert budget <= token_limit
    if maximum_context_length is not None:
        assert budget <= maximum_context_length
