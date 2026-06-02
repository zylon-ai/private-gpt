import ast
import contextlib
import json
import logging
import math
from typing import Any

from injector import singleton

from private_gpt.components.chat.models.chat_config_models import ToolSpec
from private_gpt.components.context.models.context_layer import ToolDefinitionsLayer
from private_gpt.components.context.models.layer_type import LayerType
from private_gpt.components.engines.chat_loop.interceptors.chat_loop_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_interceptor_context import (
    ChatLoopInterceptorContext,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_phase import (
    InterceptorPhase,
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


def _resolve_types(prop_schema: dict[str, Any]) -> list[str]:
    raw = prop_schema.get("type")
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        return [raw]
    for union_key in ("anyOf", "oneOf"):
        if union_key in prop_schema:
            resolved: list[str] = []
            for entry in prop_schema[union_key]:
                entry_type = entry.get("type")
                if isinstance(entry_type, str):
                    resolved.append(entry_type)
                elif isinstance(entry_type, list):
                    resolved.extend(entry_type)
            return resolved
    return []


def _item_schema(prop_schema: dict[str, Any]) -> dict[str, Any]:
    return prop_schema.get("items") or {}


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
    """Parse a string as JSON, falling back to ast.literal_eval for Python repr.

    Returns the parsed value if it is an instance of `expected`, else None.
    """
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
    if not isinstance(parsed, (list, tuple)):
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
    if "properties" in prop_schema:
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
                if isinstance(value, (int, float)):
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

    if "array" in effective_types:
        return _coerce_array(key, value, prop_schema, strict)

    if "object" in effective_types:
        return _coerce_object(key, value, prop_schema, strict)

    if not effective_types:
        return value

    return _coerce_scalar(key, value, effective_types, nullable, strict)


def _coerce_kwargs(
    kwargs: dict[str, Any],
    input_schema: dict[str, Any],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    properties: dict[str, dict[str, Any]] = input_schema.get("properties") or {}
    required: set[str] = set(input_schema.get("required") or [])
    coerced: dict[str, Any] = {}

    for key, prop_schema in properties.items():
        types = _resolve_types(prop_schema)
        nullable = "null" in types

        if key not in kwargs:
            if key in required:
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
class SchemaCoercingToolInterceptor(ChatRequestLoopInterceptor):
    """Coerce tool kwargs to match the declared input_schema before invocation."""

    async def intercept(self, context: ChatLoopInterceptorContext) -> None:
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        state = context.state
        tools = [
            self._patch_tool(tool) for tool in state.input.context_stack.all_tools()
        ]

        stack = state.input.context_stack.remove_layers_of_type(
            LayerType.TOOL_DEFINITIONS
        )
        if tools:
            stack = stack.append_layer(
                ToolDefinitionsLayer(tools=tools, source="schema_coercion_patch")
            )
        state.input.context_stack = stack
        context.set_state(state)

    def _patch_tool(self, tool: ToolSpec) -> ToolSpec:
        schema = tool.input_schema or {}
        original_fn = tool.async_fn

        async def _coerced_fn(*args: Any, **kwargs: Any) -> Any:
            try:
                fixed_kwargs = _coerce_kwargs(kwargs, input_schema=schema)
            except SchemaCoercionError:
                raise
            except Exception as e:
                logger.exception(
                    "Schema coercion failed for tool '%s', invoking with original kwargs",
                    tool.name,
                    exc_info=e,
                )
                fixed_kwargs = kwargs
            return await original_fn(*args, **fixed_kwargs)

        patched = tool.model_copy(deep=True)
        patched.async_fn = _coerced_fn
        return patched
