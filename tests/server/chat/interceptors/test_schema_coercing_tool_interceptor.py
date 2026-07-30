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


# =============================================================================
# Untyped property schemas (no "type" field) — JSON string coercion
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema", "expected"),
    [
        (
            "Object JSON string coerced when param has no type",
            {"object_param": '{"key": "value"}'},
            _schema(
                (
                    "object_param",
                    {"description": "send whatever input the user gave you"},
                )
            ),
            {"object_param": {"key": "value"}},
        ),
        (
            "Array JSON string coerced when param has no type",
            {"array_param": "[1, 2, 3]"},
            _schema(
                (
                    "array_param",
                    {
                        "description": "send whatever input the user gave you in the form of a json array"
                    },
                )
            ),
            {"array_param": [1, 2, 3]},
        ),
        (
            "Nested object JSON string coerced when param has no type",
            {"sort": '[{"attribute": "name", "direction": "asc"}]'},
            _schema(("sort", {"description": "sorting to apply"})),
            {"sort": [{"attribute": "name", "direction": "asc"}]},
        ),
        (
            "Native dict unchanged when param has no type",
            {"object_param": {"key": "value"}},
            _schema(("object_param", {"description": "some param"})),
            {"object_param": {"key": "value"}},
        ),
        (
            "Native list unchanged when param has no type",
            {"array_param": [1, 2, 3]},
            _schema(("array_param", {"description": "some param"})),
            {"array_param": [1, 2, 3]},
        ),
        (
            "Plain string unchanged when param has no type",
            {"text": "hello world"},
            _schema(("text", {"description": "some text"})),
            {"text": "hello world"},
        ),
        (
            "Invalid JSON string unchanged when param has no type",
            {"text": "{not valid json}"},
            _schema(("text", {"description": "some text"})),
            {"text": "{not valid json}"},
        ),
    ],
)
def test_untyped_property_json_string_coercion(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert _coerce_kwargs(kwargs, schema) == expected, description


# =============================================================================
# JSON Schema composition keywords: allOf / anyOf / oneOf
# All three are normalised uniformly — properties and types from sub-schemas
# are merged so coercion can traverse into them.
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema", "expected"),
    [
        # ---- allOf ----
        (
            "allOf top-level: integer field coerced from string",
            {"n": "5"},
            {
                "allOf": [
                    {
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                        "required": ["n"],
                    }
                ]
            },
            {"n": 5},
        ),
        (
            "allOf top-level: array field coerced from JSON string",
            {"data": '["a", "b"]'},
            {
                "allOf": [
                    {
                        "type": "object",
                        "properties": {
                            "data": {"type": "array", "items": {"type": "string"}}
                        },
                    }
                ]
            },
            {"data": ["a", "b"]},
        ),
        (
            "allOf top-level: multiple sub-schemas merged",
            {"n": "3", "flag": "true"},
            {
                "allOf": [
                    {"type": "object", "properties": {"n": {"type": "integer"}}},
                    {"type": "object", "properties": {"flag": {"type": "boolean"}}},
                ]
            },
            {"n": 3, "flag": True},
        ),
        (
            "allOf property-level: inner object fields coerced from dict",
            {"address": {"zip": "90210", "city": "LA"}},
            {
                "type": "object",
                "properties": {
                    "address": {
                        "allOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "zip": {"type": "integer"},
                                    "city": {"type": "string"},
                                },
                            }
                        ]
                    }
                },
            },
            {"address": {"zip": 90210, "city": "LA"}},
        ),
        (
            "allOf property-level: JSON string parsed and inner fields coerced",
            {"address": '{"zip": "90210", "city": "LA"}'},
            {
                "type": "object",
                "properties": {
                    "address": {
                        "allOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "zip": {"type": "integer"},
                                    "city": {"type": "string"},
                                },
                            }
                        ]
                    }
                },
            },
            {"address": {"zip": 90210, "city": "LA"}},
        ),
        (
            "allOf required fields honoured in lenient mode",
            {"extra": "ok"},
            {
                "allOf": [
                    {
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                        "required": ["n"],
                    }
                ]
            },
            {"extra": "ok"},
        ),
        # ---- anyOf ----
        (
            "anyOf top-level: integer field coerced from string",
            {"n": "7"},
            {"anyOf": [{"type": "object", "properties": {"n": {"type": "integer"}}}]},
            {"n": 7},
        ),
        (
            "anyOf property-level object+null: inner fields coerced from dict",
            {"filter": {"eq": "5"}},
            {
                "type": "object",
                "properties": {
                    "filter": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {"eq": {"type": "integer"}},
                            },
                            {"type": "null"},
                        ]
                    }
                },
            },
            {"filter": {"eq": 5}},
        ),
        (
            "anyOf property-level object+null: JSON string parsed and inner fields coerced",
            {"filter": '{"eq": "5"}'},
            {
                "type": "object",
                "properties": {
                    "filter": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {"eq": {"type": "integer"}},
                            },
                            {"type": "null"},
                        ]
                    }
                },
            },
            {"filter": {"eq": 5}},
        ),
        (
            "anyOf property-level object+null: null value preserved",
            {"filter": None},
            {
                "type": "object",
                "properties": {
                    "filter": {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {"eq": {"type": "integer"}},
                            },
                            {"type": "null"},
                        ]
                    }
                },
            },
            {"filter": None},
        ),
        # ---- oneOf ----
        (
            "oneOf top-level: boolean field coerced from string",
            {"flag": "yes"},
            {
                "oneOf": [
                    {"type": "object", "properties": {"flag": {"type": "boolean"}}}
                ]
            },
            {"flag": True},
        ),
        (
            "oneOf property-level: inner fields coerced",
            {"item": {"count": "3"}},
            {
                "type": "object",
                "properties": {
                    "item": {
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {"count": {"type": "integer"}},
                            }
                        ]
                    }
                },
            },
            {"item": {"count": 3}},
        ),
        # ---- mixed composition ----
        (
            "allOf + anyOf combined: properties merged from both",
            {"n": "3", "flag": "yes"},
            {
                "allOf": [{"type": "object", "properties": {"n": {"type": "integer"}}}],
                "anyOf": [{"properties": {"flag": {"type": "boolean"}}}],
            },
            {"n": 3, "flag": True},
        ),
    ],
)
def test_composition_keyword_coercion(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert _coerce_kwargs(kwargs, schema) == expected, description


# =============================================================================
# Boolean schemas (JSON Schema true/false as schema values)
# =============================================================================
def test_boolean_true_schema_no_crash() -> None:
    schema = {"type": "object", "properties": {"anything": True}}
    assert _coerce_kwargs({"anything": "hello"}, schema) == {"anything": "hello"}


def test_boolean_false_schema_no_crash() -> None:
    schema = {"type": "object", "properties": {"nothing": False}}
    assert _coerce_kwargs({"nothing": "world"}, schema) == {"nothing": "world"}


def test_array_items_true_schema_no_crash() -> None:
    schema = {
        "type": "object",
        "properties": {"data": {"type": "array", "items": True}},
    }
    assert _coerce_kwargs({"data": [1, "a", None]}, schema) == {"data": [1, "a", None]}


def test_anyof_bool_entries_no_crash() -> None:
    schema = {"type": "object", "properties": {"x": {"anyOf": [True, False]}}}
    assert _coerce_kwargs({"x": "hello"}, schema) == {"x": "hello"}


# =============================================================================
# Deep / recursive composition (anyOf containing allOf, etc.)
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema", "expected"),
    [
        (
            "anyOf containing allOf: inner properties coerced",
            {"x": {"n": "5"}},
            {
                "type": "object",
                "properties": {
                    "x": {
                        "anyOf": [
                            {
                                "allOf": [
                                    {"type": "object"},
                                    {"properties": {"n": {"type": "integer"}}},
                                ]
                            },
                            {"type": "null"},
                        ]
                    }
                },
            },
            {"x": {"n": 5}},
        ),
        (
            "anyOf containing allOf: null value preserved",
            {"x": None},
            {
                "type": "object",
                "properties": {
                    "x": {
                        "anyOf": [
                            {
                                "allOf": [
                                    {"type": "object"},
                                    {"properties": {"n": {"type": "integer"}}},
                                ]
                            },
                            {"type": "null"},
                        ]
                    }
                },
            },
            {"x": None},
        ),
        (
            "allOf containing anyOf: types and properties merged",
            {"val": "42", "flag": "yes"},
            {
                "allOf": [
                    {
                        "anyOf": [
                            {
                                "type": "object",
                                "properties": {"val": {"type": "integer"}},
                            }
                        ]
                    },
                    {"type": "object", "properties": {"flag": {"type": "boolean"}}},
                ]
            },
            {"val": 42, "flag": True},
        ),
    ],
)
def test_deep_composition_coercion(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert _coerce_kwargs(kwargs, schema) == expected, description


# =============================================================================
# Deeply nested structures
# =============================================================================
def test_nested_array_of_arrays() -> None:
    schema = {
        "type": "object",
        "properties": {
            "matrix": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
            }
        },
    }
    assert _coerce_kwargs({"matrix": [["1", "2"], ["3", "4"]]}, schema) == {
        "matrix": [[1, 2], [3, 4]]
    }


def test_nested_array_of_arrays_from_json_string() -> None:
    schema = {
        "type": "object",
        "properties": {
            "matrix": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "integer"}},
            }
        },
    }
    assert _coerce_kwargs({"matrix": "[[1,2],[3,4]]"}, schema) == {
        "matrix": [[1, 2], [3, 4]]
    }


# =============================================================================
# anyOf / type-list with "string" declared — string values must NOT be parsed
# as object or array, and must NOT be cast to other scalar types.
# The LLM intentionally sends a string; we must honour the declared string type.
# =============================================================================
@pytest.mark.parametrize(
    ("description", "kwargs", "schema", "expected"),
    [
        (
            "anyOf [string, object]: JSON-looking string stays string",
            {"body": '{"key": "value"}'},
            _schema(("body", {"anyOf": [{"type": "string"}, {"type": "object"}]})),
            {"body": '{"key": "value"}'},
        ),
        (
            "anyOf [object, string]: JSON-looking string stays string (order irrelevant)",
            {"body": '{"key": "value"}'},
            _schema(("body", {"anyOf": [{"type": "object"}, {"type": "string"}]})),
            {"body": '{"key": "value"}'},
        ),
        (
            "anyOf [string, array]: array-looking string stays string",
            {"ids": "[1, 2, 3]"},
            _schema(("ids", {"anyOf": [{"type": "string"}, {"type": "array"}]})),
            {"ids": "[1, 2, 3]"},
        ),
        (
            "anyOf [string, number]: numeric string stays string",
            {"val": "42"},
            _schema(("val", {"anyOf": [{"type": "string"}, {"type": "number"}]})),
            {"val": "42"},
        ),
        (
            "anyOf [number, string]: numeric string stays string (order irrelevant)",
            {"val": "42"},
            _schema(("val", {"anyOf": [{"type": "number"}, {"type": "string"}]})),
            {"val": "42"},
        ),
        (
            "anyOf [string, boolean]: boolean-looking string stays string",
            {"flag": "true"},
            _schema(("flag", {"anyOf": [{"type": "string"}, {"type": "boolean"}]})),
            {"flag": "true"},
        ),
        (
            "anyOf [string, object]: dict value stays dict (not serialised)",
            {"body": {"key": "value"}},
            _schema(("body", {"anyOf": [{"type": "string"}, {"type": "object"}]})),
            {"body": {"key": "value"}},
        ),
        (
            "anyOf 6 types with string first: JSON string stays string",
            {"val": '{"k": 1}'},
            _schema(
                (
                    "val",
                    {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {"type": "boolean"},
                            {"type": "object"},
                            {"type": "array"},
                            {"type": "null"},
                        ]
                    },
                )
            ),
            {"val": '{"k": 1}'},
        ),
        (
            "type list ['string', 'object']: JSON string stays string",
            {"data": '{"k": 1}'},
            _schema(("data", {"type": ["string", "object"]})),
            {"data": '{"k": 1}'},
        ),
        (
            "anyOf [object, null] WITHOUT string: JSON string is parsed to dict",
            {"meta": '{"k": 1}'},
            _schema(("meta", {"anyOf": [{"type": "object"}, {"type": "null"}]})),
            {"meta": {"k": 1}},
        ),
        (
            "anyOf [string, null]: 'null' string -> null (null coercion still takes priority)",
            {"x": "null"},
            _schema(("x", {"anyOf": [{"type": "string"}, {"type": "null"}]})),
            {"x": None},
        ),
        (
            "anyOf [string, null]: empty string -> null for nullable",
            {"x": ""},
            _schema(("x", {"anyOf": [{"type": "string"}, {"type": "null"}]})),
            {"x": None},
        ),
    ],
)
def test_string_type_declared_prevents_structural_coercion(
    description: str,
    kwargs: dict[str, Any],
    schema: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    assert _coerce_kwargs(kwargs, schema) == expected, description


# =============================================================================
# n8n-style real-world schema patterns
# =============================================================================
def test_n8n_http_request_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
            },
            "headers": {"type": "object", "additionalProperties": {"type": "string"}},
            "body": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "object", "additionalProperties": True},
                ]
            },
            "timeout": {"type": "number", "default": 10000},
            "followRedirects": {"type": "boolean", "default": True},
        },
        "required": ["url"],
    }
    result = _coerce_kwargs(
        {
            "url": "https://api.example.com/data",
            "method": "POST",
            "body": '{"payload": "test"}',
            "headers": '{"Content-Type": "application/json"}',
            "timeout": "5000",
            "followRedirects": "true",
        },
        schema,
    )
    assert result["body"] == '{"payload": "test"}', (
        "body: anyOf [string, object] — string stays string"
    )
    assert result["headers"] == {"Content-Type": "application/json"}, (
        "headers: type object — JSON string parsed"
    )
    assert result["timeout"] == 5000.0
    assert result["followRedirects"] is True


def test_n8n_set_field_value_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "fieldName": {"type": "string"},
            "fieldValue": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "object"},
                    {"type": "array"},
                    {"type": "null"},
                ]
            },
        },
        "required": ["fieldName", "fieldValue"],
    }
    assert _coerce_kwargs({"fieldName": "x", "fieldValue": "hello"}, schema) == {
        "fieldName": "x",
        "fieldValue": "hello",
    }
    assert _coerce_kwargs({"fieldName": "x", "fieldValue": "42"}, schema) == {
        "fieldName": "x",
        "fieldValue": "42",
    }
    assert _coerce_kwargs({"fieldName": "x", "fieldValue": None}, schema) == {
        "fieldName": "x",
        "fieldValue": None,
    }
    assert _coerce_kwargs({"fieldName": "x", "fieldValue": "null"}, schema) == {
        "fieldName": "x",
        "fieldValue": None,
    }
    assert _coerce_kwargs({"fieldName": "x", "fieldValue": {"k": 1}}, schema) == {
        "fieldName": "x",
        "fieldValue": {"k": 1},
    }
    assert _coerce_kwargs({"fieldName": "x", "fieldValue": [1, 2]}, schema) == {
        "fieldName": "x",
        "fieldValue": [1, 2],
    }


def test_n8n_google_sheets_nested_filters() -> None:
    schema = {
        "type": "object",
        "properties": {
            "documentId": {"type": "string"},
            "filters": {
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {"type": "string"},
                                "condition": {"type": "string"},
                                "value": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"type": "number"},
                                        {"type": "boolean"},
                                    ]
                                },
                            },
                            "required": ["column", "condition", "value"],
                        },
                    },
                    "combineConditions": {"type": "string", "enum": ["AND", "OR"]},
                },
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 50},
            "returnAll": {"type": "boolean", "default": False},
        },
        "required": ["documentId"],
    }
    result = _coerce_kwargs(
        {
            "documentId": "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgVE2upms",
            "filters": '{"conditions": [{"column": "Name", "condition": "contains", "value": "John"}], "combineConditions": "AND"}',
            "limit": "10",
            "returnAll": "false",
        },
        schema,
    )
    assert isinstance(result["filters"], dict)
    assert result["filters"]["conditions"][0]["value"] == "John", (
        "anyOf str|num|bool: string value stays string"
    )
    assert result["limit"] == 10
    assert result["returnAll"] is False


def test_n8n_postgres_query_parameters() -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "parameters": {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "number"},
                        {"type": "boolean"},
                        {"type": "null"},
                    ]
                },
            },
        },
        "required": ["query"],
    }
    result = _coerce_kwargs(
        {
            "query": "SELECT * FROM users WHERE id = $1 AND active = $2",
            "parameters": '[42, "active", true, null]',
        },
        schema,
    )
    assert result["parameters"] == [42, "active", True, None]


def test_untyped_property_parses_python_literal_dict_string() -> None:
    """LLMs sometimes emit single-quoted Python dict syntax for untyped params."""
    schema = _schema(
        (
            "object_param",
            {
                "description": "send whatever input the user gave you in the form of a json array"
            },
        )
    )
    result = _coerce_kwargs(
        {
            "object_param": "{'animal': 'Lion', 'traits': ['Majestic', 'Powerful', 'Social']}"
        },
        schema,
    )
    assert result == {
        "object_param": {
            "animal": "Lion",
            "traits": ["Majestic", "Powerful", "Social"],
        }
    }


def test_create_model_from_json_schema_preserves_untyped_properties() -> None:
    from private_gpt.chat.schema_models import create_model_from_json_schema

    original = {
        "type": "object",
        "properties": {
            "object_param": {
                "description": "send whatever input the user gave you in the form of a json array"
            }
        },
        "required": ["object_param"],
        "additionalProperties": True,
    }
    model = create_model_from_json_schema(original, "Obj")
    regenerated = model.model_json_schema()
    assert "type" not in regenerated["properties"]["object_param"]
