from __future__ import annotations

import ast
import contextlib
import json
import logging
import math
from typing import TYPE_CHECKING, Any

from injector import singleton

from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_phase import (
    InterceptorPhase,
)
from private_gpt.components.tools.remote_execution import (
    ToolExecutionInterceptor,
    ToolExecutionInterceptorContext,
)

if TYPE_CHECKING:
    from private_gpt.components.engines.chat.models.chat_interceptor_context import (
        ChatInterceptorContext,
    )

logger = logging.getLogger(__name__)

_SCALAR_COERCIONS: dict[str, type] = {
    "integer": int,
    "number": float,
    "boolean": bool,
    "string": str,
}


_NULL_STRINGS: frozenset[str] = frozenset({"null", "none", "nil", "undefined"})
_TRUE_STRINGS: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_FALSE_STRINGS: frozenset[str] = frozenset({"false", "0", "no", "off"})

# JSON Schema composition keywords whose sub-schemas are flattened during coercion.
# For coercion purposes all three are treated identically: collect every possible
# type and merge all declared properties (best-effort — we want to coerce as much
# as possible without rejecting valid values).
_COMPOSITION_KEYS: tuple[str, ...] = ("allOf", "anyOf", "oneOf")


class SchemaCoercionError(ValueError):
    """Raised in strict mode when a value cannot be coerced."""

    def __init__(self, key: str, value: Any, expected: str) -> None:
        super().__init__(f"Cannot coerce param '{key}'={value!r} to {expected}")
        self.key = key
        self.value = value
        self.expected = expected


def _type_name(expected: type | tuple[type, ...]) -> str:
    if isinstance(expected, tuple):
        return "/".join(t.__name__ for t in expected)
    return expected.__name__


class _NormalizedSchema:
    """Flat view of a JSON Schema after resolving composition keywords.

    Handles ``allOf`` / ``anyOf`` / ``oneOf`` uniformly: all sub-schemas are
    merged so that downstream coercion helpers only ever read from a single,
    predictable structure.
    """

    __slots__ = ("items", "properties", "required", "types")

    def __init__(
        self,
        types: list[str],
        properties: dict[str, dict[str, Any]],
        required: set[str],
        items: dict[str, Any],
    ) -> None:
        self.types = types
        self.properties = properties
        self.required = required
        self.items = items


def _normalize_schema(schema: dict[str, Any]) -> _NormalizedSchema:
    """Produce a flat coercion-view of *schema*.

    Merges every sub-schema found under ``allOf``, ``anyOf``, and ``oneOf``
    into the top-level fields so callers never need to inspect composition
    keywords directly.

    Boolean schemas (JSON Schema ``true`` / ``false``) are silently skipped —
    ``true`` means "accept anything" and ``false`` means "reject everything";
    neither carries structural coercion information.
    """
    # Boolean schemas (JSON Schema spec allows true/false as valid schemas).
    if not isinstance(schema, dict):
        return _NormalizedSchema(types=[], properties={}, required=set(), items={})

    # Collect type(s) declared at the top level first.
    raw_type = schema.get("type")
    if isinstance(raw_type, list):
        types: list[str] = [t for t in raw_type if isinstance(t, str)]
    elif isinstance(raw_type, str):
        types = [raw_type]
    else:
        types = []

    properties: dict[str, dict[str, Any]] = dict(schema.get("properties") or {})
    required: set[str] = set(schema.get("required") or [])
    items: dict[str, Any] = schema.get("items") or {}

    # Flatten composition keywords recursively — order: allOf, anyOf, oneOf.
    # Each sub-schema is itself normalised first so that nested composition
    # (e.g. anyOf containing allOf) is resolved transitively.
    for key in _COMPOSITION_KEYS:
        for entry in schema.get(key) or []:
            # Skip boolean sub-schemas.
            if not isinstance(entry, dict):
                continue

            sub = _normalize_schema(entry)

            # Merge types (union — broadest possible for coercion).
            for t in sub.types:
                if t not in types:
                    types.append(t)

            # Merge properties — top-level schema wins on conflicts.
            for prop_key, prop_schema in sub.properties.items():
                if prop_key not in properties:
                    properties[prop_key] = prop_schema

            # Merge required fields.
            required.update(sub.required)

            # Inherit items schema if not already set.
            if not items and sub.items:
                items = sub.items

    return _NormalizedSchema(
        types=types,
        properties=properties,
        required=required,
        items=items,
    )


def _resolve_types(prop_schema: dict[str, Any]) -> list[str]:
    return _normalize_schema(prop_schema).types


def _item_schema(prop_schema: dict[str, Any]) -> dict[str, Any]:
    return _normalize_schema(prop_schema).items


def _is_non_finite_float(value: Any) -> bool:
    return (
        isinstance(value, float)
        and not isinstance(value, bool)
        and (math.isnan(value) or math.isinf(value))
    )


def _parse_literal_string(
    key: str,
    raw: str,
    expected: type | tuple[type, ...],
) -> Any:
    with contextlib.suppress(json.JSONDecodeError, ValueError):
        parsed = json.loads(raw)
        if isinstance(parsed, expected):
            return parsed
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError, MemoryError, RecursionError, TypeError):
        logger.warning(
            "Failed to parse string as %s for param '%s'",
            _type_name(expected),
            key,
        )
        return None
    if isinstance(parsed, expected):
        return parsed
    return None


def _coerce_array(
    key: str,
    value: Any,
    prop_schema: dict[str, Any],
    strict: bool,
) -> Any:
    parsed: Any = value
    if isinstance(value, str):
        parsed = _parse_literal_string(key, value, (list, tuple))
    if not isinstance(parsed, list | tuple):
        if strict:
            raise SchemaCoercionError(key, value, "array")
        return value

    items = list(parsed)
    item_schema = _item_schema(prop_schema)
    if not item_schema:
        return items
    return [
        _coerce_value(f"{key}[{i}]", v, item_schema, strict)
        for i, v in enumerate(items)
    ]


def _coerce_object(
    key: str,
    value: Any,
    prop_schema: dict[str, Any],
    strict: bool,
) -> Any:
    parsed: dict[str, Any] | None
    if isinstance(value, dict):
        parsed = value
    elif isinstance(value, str):
        parsed = _parse_literal_string(key, value, dict)
    else:
        parsed = None

    if not isinstance(parsed, dict):
        if strict:
            raise SchemaCoercionError(key, value, "object")
        return value

    norm = _normalize_schema(prop_schema)
    if norm.properties:
        return _coerce_kwargs(parsed, prop_schema, strict=strict)
    return parsed


def _coerce_scalar(
    key: str,
    value: Any,
    effective_types: list[str],
    nullable: bool,
    strict: bool,
) -> Any:
    for scalar_type, caster in _SCALAR_COERCIONS.items():
        if scalar_type not in effective_types:
            continue
        if isinstance(value, caster) and not (
            scalar_type == "integer" and isinstance(value, bool)
        ):
            if scalar_type == "number" and _is_non_finite_float(value):
                if nullable:
                    return None
                if strict:
                    raise SchemaCoercionError(key, value, "finite number")
                return value
            return value
        try:
            if scalar_type == "boolean":
                if isinstance(value, str):
                    lower = value.strip().lower()
                    if lower in _TRUE_STRINGS:
                        return True
                    if lower in _FALSE_STRINGS:
                        return False
                    continue
                if isinstance(value, int | float):
                    return bool(value)
                continue
            if scalar_type == "integer":
                if isinstance(value, bool):
                    return int(value)
                if isinstance(value, float):
                    return int(value)
                if isinstance(value, str):
                    try:
                        return int(float(value.strip()))
                    except (ValueError, TypeError):
                        continue
            if scalar_type in ("integer", "number"):
                coerced = caster(value.strip() if isinstance(value, str) else value)
                if _is_non_finite_float(coerced):
                    if nullable:
                        return None
                    if strict:
                        raise SchemaCoercionError(key, value, "finite number")
                    return coerced
                return coerced
            return caster(value)
        except (ValueError, TypeError):
            logger.warning("Failed to coerce param '%s' to %s", key, scalar_type)

    if strict and effective_types:
        raise SchemaCoercionError(key, value, " or ".join(effective_types))
    return value


def _coerce_value(
    key: str,
    value: Any,
    prop_schema: dict[str, Any],
    strict: bool,
) -> Any:
    types = _resolve_types(prop_schema)
    nullable = "null" in types
    effective_types = [t for t in types if t != "null"]

    if value is None:
        if nullable:
            return None
        if strict:
            raise SchemaCoercionError(
                key, value, " or ".join(effective_types) or "non-null"
            )
        return value

    if _is_non_finite_float(value) and nullable:
        return None

    if isinstance(value, str):
        stripped = value.strip()
        lowered = stripped.lower()
        if lowered in _NULL_STRINGS and nullable:
            return None
        if stripped == "" and nullable:
            return None

    # If the value is a string and "string" is one of the declared types, it is already
    # valid — return it immediately without trying to coerce or parse it.
    # This prevents e.g. '{"k":1}' being parsed to a dict when the schema is
    # anyOf [string, object], and "42" being cast to float when schema is
    # anyOf [string, number].
    if isinstance(value, str) and "string" in effective_types:
        return value

    # A bool value is already valid for "boolean" — skip further dispatch.
    if isinstance(value, bool) and "boolean" in effective_types:
        return value

    if "array" in effective_types:
        return _coerce_array(key, value, prop_schema, strict)

    if "object" in effective_types:
        return _coerce_object(key, value, prop_schema, strict)

    if not effective_types:
        # Free-form / untyped property: parse JSON-looking strings into native
        # structures. Prefer JSON, then Python literals (LLMs sometimes emit
        # single-quoted dict/list syntax).
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith(("{", "[")):
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    return json.loads(stripped)
                parsed = _parse_literal_string(key, stripped, (dict, list, tuple))
                if parsed is not None:
                    return list(parsed) if isinstance(parsed, tuple) else parsed
        return value

    return _coerce_scalar(key, value, effective_types, nullable, strict)


def _coerce_kwargs(
    kwargs: dict[str, Any],
    input_schema: dict[str, Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    norm = _normalize_schema(input_schema)
    coerced: dict[str, Any] = {}

    for key, prop_schema in norm.properties.items():
        types = _resolve_types(prop_schema)
        nullable = "null" in types

        if key not in kwargs:
            if key in norm.required:
                if strict:
                    raise SchemaCoercionError(key, None, "required field missing")
                continue
            if nullable:
                coerced[key] = None
            continue

        coerced[key] = _coerce_value(key, kwargs[key], prop_schema, strict)

    for key, value in kwargs.items():
        if key not in coerced:
            coerced[key] = value

    return coerced


@singleton
class SchemaCoercingToolInterceptor(
    ChatRequestLoopInterceptor,
    ToolExecutionInterceptor,
):
    """Coerce tool kwargs to the declared schema before execution."""

    async def intercept(
        self,
        context: ChatInterceptorContext | ToolExecutionInterceptorContext,
    ) -> None:
        if not isinstance(context, ToolExecutionInterceptorContext):
            return
        if context.phase != InterceptorPhase.BEFORE_TOOL:
            return

        schema = context.request.tool_spec.input_schema or {}
        try:
            context.set_tool_kwargs(
                _coerce_kwargs(context.tool_kwargs, input_schema=schema)
            )
        except SchemaCoercionError:
            raise
        except Exception as e:
            logger.exception(
                "Schema coercion failed for tool '%s', invoking with original kwargs",
                context.request.tool_spec.name,
                exc_info=e,
            )
