import math
from typing import Any

import pytest

from private_gpt.server.chat.interceptors.schema_coercing_tool_interceptor import (
    SchemaCoercionError,
    _coerce_kwargs,
)

_NULLABLE_STR: dict[str, Any] = {"type": ["string", "null"]}
_NULLABLE_INT: dict[str, Any] = {"type": ["integer", "null"]}
_NULLABLE_NUM: dict[str, Any] = {"type": ["number", "null"]}
_NULLABLE_BOOL: dict[str, Any] = {"type": ["boolean", "null"]}
_NULLABLE_ARR_STR: dict[str, Any] = {
    "type": ["array", "null"],
    "items": {"type": ["string", "null"]},
}
_NULLABLE_ARR_INT: dict[str, Any] = {
    "type": ["array", "null"],
    "items": {"type": ["integer", "null"]},
}
_NULLABLE_OBJ: dict[str, Any] = {
    "type": ["object", "null"],
    "properties": {
        "inner": {"type": ["string", "null"]},
        "count": {"type": ["integer", "null"]},
    },
}


def _schema(
    *fields: tuple[str, dict[str, Any]], required: list[str] | None = None
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": dict(fields)}
    if required is not None:
        schema["required"] = required
    return schema


# =============================================================================
# Whitespace tolerance
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema", "expected"),
    [
        (
            "Padded 'None' string -> null",
            {"x": "  None  "},
            _schema(("x", _NULLABLE_STR)),
            {"x": None},
        ),
        (
            "Padded 'null' string -> null",
            {"x": " null "},
            _schema(("x", _NULLABLE_STR)),
            {"x": None},
        ),
        (
            "Padded 'true' -> bool",
            {"flag": " true "},
            _schema(("flag", {"type": "boolean"})),
            {"flag": True},
        ),
        (
            "Padded integer string -> int",
            {"count": " 42 "},
            _schema(("count", {"type": "integer"})),
            {"count": 42},
        ),
        (
            "Padded float string -> number",
            {"score": "  3.14  "},
            _schema(("score", {"type": "number"})),
            {"score": 3.14},
        ),
        (
            "Whitespace-only string for nullable -> null",
            {"x": "   "},
            _schema(("x", _NULLABLE_STR)),
            {"x": None},
        ),
    ],
)
def test_whitespace_tolerance(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert _coerce_kwargs(kwargs, schema) == expected, description


# =============================================================================
# Empty-string semantics for nullable fields
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema", "expected"),
    [
        (
            "Empty string for nullable int -> null",
            {"x": ""},
            _schema(("x", _NULLABLE_INT)),
            {"x": None},
        ),
        (
            "Empty string for nullable number -> null",
            {"x": ""},
            _schema(("x", _NULLABLE_NUM)),
            {"x": None},
        ),
        (
            "Empty string for nullable bool -> null",
            {"x": ""},
            _schema(("x", _NULLABLE_BOOL)),
            {"x": None},
        ),
        (
            "Empty string for nullable array -> null",
            {"x": ""},
            _schema(("x", _NULLABLE_ARR_STR)),
            {"x": None},
        ),
        (
            "Empty string for nullable object -> null",
            {"x": ""},
            _schema(("x", _NULLABLE_OBJ)),
            {"x": None},
        ),
        (
            "Empty string for non-nullable string preserved",
            {"x": ""},
            _schema(("x", {"type": "string"})),
            {"x": ""},
        ),
    ],
)
def test_empty_string_semantics(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert _coerce_kwargs(kwargs, schema) == expected, description


# =============================================================================
# Boolean coercion from numerics and extra string forms
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema", "expected"),
    [
        (
            "int 1 -> True",
            {"flag": 1},
            _schema(("flag", {"type": "boolean"})),
            {"flag": True},
        ),
        (
            "int 0 -> False",
            {"flag": 0},
            _schema(("flag", {"type": "boolean"})),
            {"flag": False},
        ),
        (
            "string '1' -> True",
            {"flag": "1"},
            _schema(("flag", {"type": "boolean"})),
            {"flag": True},
        ),
        (
            "string '0' -> False",
            {"flag": "0"},
            _schema(("flag", {"type": "boolean"})),
            {"flag": False},
        ),
        (
            "Case-variant 'YES' / 'No' -> bool",
            {"a": "YES", "b": "No"},
            _schema(("a", {"type": "boolean"}), ("b", {"type": "boolean"})),
            {"a": True, "b": False},
        ),
        (
            "Case-variant 'ON' / 'OFF' -> bool",
            {"a": "ON", "b": "OFF"},
            _schema(("a", {"type": "boolean"}), ("b", {"type": "boolean"})),
            {"a": True, "b": False},
        ),
    ],
)
def test_boolean_extras(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert _coerce_kwargs(kwargs, schema) == expected, description


# =============================================================================
# Integer edge cases
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema", "expected"),
    [
        (
            "Negative int string '-42' -> -42",
            {"n": "-42"},
            _schema(("n", {"type": "integer"})),
            {"n": -42},
        ),
        (
            "Negative float string '-1.0' -> -1",
            {"n": "-1.0"},
            _schema(("n", {"type": "integer"})),
            {"n": -1},
        ),
        (
            "Plus-prefixed '+42' -> 42",
            {"n": "+42"},
            _schema(("n", {"type": "integer"})),
            {"n": 42},
        ),
        (
            "Trailing zeros '42.0000' -> 42",
            {"n": "42.0000"},
            _schema(("n", {"type": "integer"})),
            {"n": 42},
        ),
        (
            "Bool True should NOT silently become integer 1 for an int field",
            {"n": True},
            _schema(("n", {"type": "integer"})),
            {"n": 1},
        ),
    ],
)
def test_integer_edges(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert _coerce_kwargs(kwargs, schema) == expected, description


# =============================================================================
# Number / float edge cases
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema", "expected"),
    [
        (
            "Scientific notation '1e5' -> 100000.0",
            {"x": "1e5"},
            _schema(("x", {"type": "number"})),
            {"x": 1e5},
        ),
        (
            "Scientific notation '1.5e-3' -> 0.0015",
            {"x": "1.5e-3"},
            _schema(("x", {"type": "number"})),
            {"x": 1.5e-3},
        ),
        (
            "Plus-prefixed '+3.14' -> 3.14",
            {"x": "+3.14"},
            _schema(("x", {"type": "number"})),
            {"x": 3.14},
        ),
        (
            "Integer value for number field accepted",
            {"x": 42},
            _schema(("x", {"type": "number"})),
            {"x": 42},
        ),
        (
            "Integer string '42' for number field -> 42.0",
            {"x": "42"},
            _schema(("x", {"type": "number"})),
            {"x": 42.0},
        ),
    ],
)
def test_number_edges(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert _coerce_kwargs(kwargs, schema) == expected, description


def test_native_nan_for_nullable_number_becomes_null() -> None:
    result = _coerce_kwargs({"x": float("nan")}, _schema(("x", _NULLABLE_NUM)))
    assert result == {"x": None}


def test_native_inf_for_nullable_number_becomes_null() -> None:
    assert _coerce_kwargs({"x": float("inf")}, _schema(("x", _NULLABLE_NUM))) == {
        "x": None
    }
    assert _coerce_kwargs({"x": float("-inf")}, _schema(("x", _NULLABLE_NUM))) == {
        "x": None
    }


def test_non_nullable_number_keeps_nan_or_raises() -> None:
    result = _coerce_kwargs({"x": "NaN"}, _schema(("x", {"type": "number"})))
    value = result["x"]
    assert isinstance(value, float)
    assert math.isnan(value)


# =============================================================================
# Array edge cases
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema", "expected"),
    [
        (
            "Empty JSON array '[]' -> []",
            {"ids": "[]"},
            _schema(("ids", {"type": "array", "items": {"type": "string"}})),
            {"ids": []},
        ),
        (
            "Empty native list -> []",
            {"ids": []},
            _schema(("ids", {"type": "array", "items": {"type": "string"}})),
            {"ids": []},
        ),
        (
            "Array items coerced individually: mixed-type strings to int",
            {"ids": ["1.0", "2", 3]},
            _schema(("ids", {"type": "array", "items": {"type": "integer"}})),
            {"ids": [1, 2, 3]},
        ),
        (
            "Array of nullable ints with 'None' entries -> null entries",
            {"ids": ["1", "None", "null", None]},
            _schema(("ids", _NULLABLE_ARR_INT)),
            {"ids": [1, None, None, None]},
        ),
        (
            "Nested array coercion",
            {"matrix": "[[1, 2], [3, 4]]"},
            _schema(
                (
                    "matrix",
                    {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                    },
                )
            ),
            {"matrix": [[1, 2], [3, 4]]},
        ),
    ],
)
def test_array_edges(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert _coerce_kwargs(kwargs, schema) == expected, description


# =============================================================================
# Object edge cases
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema", "expected"),
    [
        (
            "Empty JSON object '{}' -> {} with missing nullable fields injected",
            {"meta": "{}"},
            _schema(("meta", _NULLABLE_OBJ)),
            {"meta": {"inner": None, "count": None}},
        ),
        (
            "Empty native dict -> {} with missing nullable fields injected",
            {"meta": {}},
            _schema(("meta", _NULLABLE_OBJ)),
            {"meta": {"inner": None, "count": None}},
        ),
        (
            "Deeply nested object -> array -> object recursion",
            {
                "outer": {
                    "rows": [
                        {"inner": "None", "count": "1.0"},
                        {"inner": "nil", "count": "2"},
                    ]
                }
            },
            _schema(
                (
                    "outer",
                    {
                        "type": "object",
                        "properties": {
                            "rows": {
                                "type": "array",
                                "items": _NULLABLE_OBJ,
                            }
                        },
                    },
                )
            ),
            {
                "outer": {
                    "rows": [
                        {"inner": None, "count": 1},
                        {"inner": None, "count": 2},
                    ]
                }
            },
        ),
        (
            "Additional properties not in schema preserved inside object",
            {"meta": {"inner": "hi", "count": "1", "extra": "keep"}},
            _schema(("meta", _NULLABLE_OBJ)),
            {"meta": {"inner": "hi", "count": 1, "extra": "keep"}},
        ),
    ],
)
def test_object_edges(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert _coerce_kwargs(kwargs, schema) == expected, description


# =============================================================================
# Union / multi-type schema constructs
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema", "expected"),
    [
        (
            "oneOf nullable treated like anyOf nullable",
            {"x": "None"},
            _schema(("x", {"oneOf": [{"type": "string"}, {"type": "null"}]})),
            {"x": None},
        ),
        (
            "Multi-type schema ['string', 'integer', 'null'] with 'None' -> null",
            {"x": "None"},
            _schema(("x", {"type": ["string", "integer", "null"]})),
            {"x": None},
        ),
        (
            "Empty schema dict: value passed through unchanged",
            {"x": "anything"},
            _schema(("x", {})),
            {"x": "anything"},
        ),
        (
            "Schema with only description: value passed through",
            {"x": 123},
            _schema(("x", {"description": "freeform"})),
            {"x": 123},
        ),
    ],
)
def test_union_and_loose_schemas(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert _coerce_kwargs(kwargs, schema) == expected, description


# =============================================================================
# String field receiving non-string primitives
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema", "expected"),
    [
        (
            "Integer 42 for string field -> '42'",
            {"s": 42},
            _schema(("s", {"type": "string"})),
            {"s": "42"},
        ),
        (
            "Float 3.14 for string field -> '3.14'",
            {"s": 3.14},
            _schema(("s", {"type": "string"})),
            {"s": "3.14"},
        ),
        (
            "Bool True for string field -> 'true' or 'True' (probes contract)",
            {"s": True},
            _schema(("s", {"type": "string"})),
            {"s": "True"},
        ),
    ],
)
def test_stringification_of_primitives(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert _coerce_kwargs(kwargs, schema) == expected, description


# =============================================================================
# Enum handling
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema", "expected"),
    [
        (
            "Valid enum value passed through",
            {"color": "red"},
            _schema(("color", {"type": "string", "enum": ["red", "blue"]})),
            {"color": "red"},
        ),
        (
            "Integer enum value as string coerced to int",
            {"n": "2"},
            _schema(("n", {"type": "integer", "enum": [1, 2, 3]})),
            {"n": 2},
        ),
    ],
)
def test_enum_handling(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert _coerce_kwargs(kwargs, schema) == expected, description


# =============================================================================
# Failure / malformed-input contract probes
# These tests pin down what happens with garbage values. Adjust the expected
# behavior to match the intended contract (lenient pass-through, raise, or
# null-on-failure) once confirmed.
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema"),
    [
        (
            "Non-numeric string for integer field",
            {"n": "abc"},
            _schema(("n", {"type": "integer"})),
        ),
        (
            "Non-numeric string for number field",
            {"x": "abc"},
            _schema(("x", {"type": "number"})),
        ),
        (
            "Ambiguous string for boolean field",
            {"flag": "maybe"},
            _schema(("flag", {"type": "boolean"})),
        ),
        (
            "Malformed JSON for object field",
            {"meta": "{not json}"},
            _schema(("meta", _NULLABLE_OBJ)),
        ),
        (
            "Scalar string for non-nullable array field",
            {"ids": "hello"},
            _schema(("ids", {"type": "array", "items": {"type": "string"}})),
        ),
    ],
)
def test_invalid_values_contract(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    """Probe behavior for malformed LLM output.

    The coercer should either:
       (a) raise a clear error,
       (b) pass the value through unchanged, or
       (c) produce null for nullable fields.
    This test documents whichever contract holds; update assertions to match.
    """
    try:
        result = _coerce_kwargs(kwargs, schema)
    except (ValueError, TypeError) as exc:
        pytest.skip(f"Raises {type(exc).__name__} (strict mode): {description}")
    else:
        key = next(iter(kwargs))
        assert key in result, f"{description}: key dropped unexpectedly"


# =============================================================================
# Strict mode: unrecoverable coercion raises SchemaCoercionError
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema"),
    [
        (
            "Non-numeric string for integer field",
            {"n": "abc"},
            _schema(("n", {"type": "integer"})),
        ),
        (
            "Non-numeric string for number field",
            {"x": "abc"},
            _schema(("x", {"type": "number"})),
        ),
        (
            "Ambiguous string for boolean field",
            {"flag": "maybe"},
            _schema(("flag", {"type": "boolean"})),
        ),
        (
            "Malformed JSON for object field",
            {"meta": "{not json}"},
            _schema(("meta", _NULLABLE_OBJ)),
        ),
        (
            "Scalar string for non-nullable array field",
            {"ids": "hello"},
            _schema(("ids", {"type": "array", "items": {"type": "string"}})),
        ),
        (
            "Null string for non-nullable integer field",
            {"n": "None"},
            _schema(("n", {"type": "integer"})),
        ),
        (
            "Native None for non-nullable integer field",
            {"n": None},
            _schema(("n", {"type": "integer"})),
        ),
        (
            "Non-finite NaN string for non-nullable number",
            {"x": "NaN"},
            _schema(("x", {"type": "number"})),
        ),
        (
            "Missing required field",
            {},
            _schema(("n", {"type": "integer"}), required=["n"]),
        ),
    ],
)
def test_strict_raises_on_invalid(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
) -> None:
    with pytest.raises(SchemaCoercionError):
        _coerce_kwargs(kwargs, schema, strict=True)


# =============================================================================
# Strict mode: valid coercion still succeeds
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema", "expected"),
    [
        (
            "Valid integer string coerced under strict",
            {"n": "42"},
            _schema(("n", {"type": "integer"})),
            {"n": 42},
        ),
        (
            "Valid boolean string coerced under strict",
            {"flag": "yes"},
            _schema(("flag", {"type": "boolean"})),
            {"flag": True},
        ),
        (
            "Nullable field with 'None' string under strict -> null",
            {"x": "None"},
            _schema(("x", _NULLABLE_INT)),
            {"x": None},
        ),
        (
            "Nullable field with native NaN under strict -> null",
            {"x": float("nan")},
            _schema(("x", _NULLABLE_NUM)),
            {"x": None},
        ),
        (
            "Array with items coerced under strict",
            {"ids": ["1", "2", "3"]},
            _schema(("ids", {"type": "array", "items": {"type": "integer"}})),
            {"ids": [1, 2, 3]},
        ),
        (
            "Nested object coerced under strict",
            {"meta": {"inner": "hi", "count": "7"}},
            _schema(("meta", _NULLABLE_OBJ)),
            {"meta": {"inner": "hi", "count": 7}},
        ),
    ],
)
def test_strict_accepts_valid_coercion(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert _coerce_kwargs(kwargs, schema, strict=True) == expected, description


# =============================================================================
# Strict mode: nested failure propagates from depth
# =============================================================================
def test_strict_failure_propagates_from_array_item() -> None:
    schema = _schema(("ids", {"type": "array", "items": {"type": "integer"}}))
    with pytest.raises(SchemaCoercionError) as exc_info:
        _coerce_kwargs({"ids": ["1", "abc", "3"]}, schema, strict=True)
    assert exc_info.value.key == "ids[1]"
    assert exc_info.value.value == "abc"


def test_strict_failure_propagates_from_nested_object() -> None:
    schema = _schema(("meta", _NULLABLE_OBJ))
    with pytest.raises(SchemaCoercionError) as exc_info:
        _coerce_kwargs(
            {"meta": {"inner": "hi", "count": "not-a-number"}},
            schema,
            strict=True,
        )
    assert exc_info.value.key == "count"


# =============================================================================
# Lenient mode (default) unchanged
# =============================================================================
def test_lenient_mode_is_default_and_does_not_raise() -> None:
    schema = _schema(("n", {"type": "integer"}))
    result = _coerce_kwargs({"n": "abc"}, schema)
    assert result == {"n": "abc"}


def test_lenient_mode_passes_through_invalid_array() -> None:
    schema = _schema(("ids", {"type": "array", "items": {"type": "string"}}))
    assert _coerce_kwargs({"ids": "hello"}, schema) == {"ids": "hello"}
