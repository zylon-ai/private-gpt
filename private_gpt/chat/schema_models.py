from collections.abc import Callable
from keyword import iskeyword, issoftkeyword
from types import UnionType
from typing import Any, Literal, Self

from pydantic import BaseModel, Field, create_model
from pydantic.config import ConfigDict, ExtraValues
from pydantic.json_schema import (
    DEFAULT_REF_TEMPLATE,
    GenerateJsonSchema,
    JsonSchemaMode,
)
from pydantic.main import IncEx

RESERVED_NAMES = frozenset(
    [
        "model_config",
        "model_fields",
        "model_computed_fields",
        "model_extra",
        "model_fields_set",
        "model_construct",
        "model_copy",
        "model_dump",
        "model_dump_json",
        "model_json_schema",
        "model_validate",
        "model_validate_json",
        "model_validate_strings",
        "model_rebuild",
        "dict",
        "json",
        "copy",
        "parse_obj",
        "parse_raw",
        "parse_file",
        "from_orm",
        "schema",
        "schema_json",
        "construct",
        "validate",
        "update_forward_refs",
    ]
)


def _sanitize_field_name(field_name: str) -> str:
    def _handle_empty_field_name(field_name: str) -> str:
        """Handle empty or None field names - fallback to 'field'."""
        return field_name if field_name.strip() else "field"

    def _handle_leading_underscores(field_name: str) -> str:
        """Handle leading underscores - Pydantic treats them as private fields."""
        return field_name.lstrip("_") or "field"

    def _handle_dunder_names(field_name: str) -> str:
        """Handle dunder names (__name__) - Python magic methods conflict."""
        if field_name.startswith("__") and field_name.endswith("__"):
            return f"dunder_{field_name[2:-2]}_field"
        return field_name

    def _handle_python_keywords(field_name: str) -> str:
        """Handle Python keywords and soft keywords - reserved language constructs."""
        if iskeyword(field_name) or issoftkeyword(field_name):
            return f"{field_name}_"
        return field_name

    def _handle_reserved_basemodel_attributes(field_name: str) -> str:
        """Handle reserved BaseModel attributes - conflicts with Pydantic internals."""
        if field_name.lower() in RESERVED_NAMES:
            return f"{field_name}_field"
        return field_name

    # Apply all sanitization steps in order
    sanitized = _handle_empty_field_name(field_name)
    sanitized = _handle_leading_underscores(sanitized)
    sanitized = _handle_dunder_names(sanitized)
    sanitized = _handle_python_keywords(sanitized)
    sanitized = _handle_reserved_basemodel_attributes(sanitized)

    return sanitized


def _validate_json_schema_item(schema: dict[str, Any], strict: bool = True) -> None:
    """Validate that the schema is a valid JSON Schema object."""
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a dictionary representing JSON Schema")

    if schema == {}:
        return

    # Handle combinators first
    if "oneOf" in schema:
        if not isinstance(schema["oneOf"], list):
            raise ValueError("'oneOf' must be an array of schemas")
        for sub_schema in schema["oneOf"]:
            _validate_json_schema_item(sub_schema, strict=False)
        return  # oneOf schemas don't need type field

    if "anyOf" in schema:
        if not isinstance(schema["anyOf"], list):
            raise ValueError("'anyOf' must be an array of schemas")
        for sub_schema in schema["anyOf"]:
            _validate_json_schema_item(sub_schema, strict=False)
        return  # anyOf schemas don't need type field

    if "allOf" in schema:
        if not isinstance(schema["allOf"], list):
            raise ValueError("'allOf' must be an array of schemas")
        for sub_schema in schema["allOf"]:
            _validate_json_schema_item(sub_schema, strict=False)
        return  # allOf schemas don't need type field

    # Regular schema validation
    if "type" not in schema:
        if strict:
            raise ValueError("Schema must define a 'type' field")
        else:
            return  # Non-strict mode allows missing type

    if schema["type"] == "array" and "items" not in schema:
        raise ValueError("Array schemas must define 'items'")

    if schema["type"] == "array" and not isinstance(schema.get("items"), dict):
        raise ValueError("Array 'items' must be a dictionary representing JSON Schema")

    # Recursively validate array items
    if schema["type"] == "array":
        _validate_json_schema_item(schema.get("items", {}))


def _validate_json_schema(schema: dict[str, Any]) -> None:
    """Validate that the schema is a valid JSON Schema object."""
    if not isinstance(schema, dict):
        raise ValueError("Schema must be a dictionary representing JSON Schema")
    if not schema:
        return

    if "type" not in schema:
        raise ValueError("Schema must define a 'type' field")

    if schema["type"] == "object":
        if "properties" not in schema:
            raise ValueError("Object schemas must define 'properties'")
        for _, prop_schema in schema.get("properties", {}).items():
            _validate_json_schema_item(prop_schema, strict=False)
    elif schema["type"] == "array":
        _validate_json_schema_item(schema.get("items", {}))
    else:
        pass


def _resolve_field_name_collisions(field_name: str, used_names: set[str]) -> str:
    """Resolve field name collisions by appending counter."""
    if field_name not in used_names:
        return field_name

    counter = 1
    while f"{field_name}_{counter}" in used_names:
        counter += 1

    return f"{field_name}_{counter}"


def _resolve_json_type(field_schema: dict[str, Any]) -> type[Any] | UnionType:
    """Resolve JSON schema type to Python type."""
    json_type_mapping = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }

    # Handle anyOf - create a Union type
    if "anyOf" in field_schema:
        types = []
        for sub_schema in field_schema["anyOf"]:
            resolved_type = _resolve_json_type(sub_schema)
            types.append(resolved_type)
        if len(types) == 1:
            return types[0]
        # Create Union type
        from typing import Union

        return Union[tuple(types)]  # type: ignore # noqa: UP007

    # Handle oneOf - similar to anyOf for typing purposes
    if "oneOf" in field_schema:
        types = []
        for sub_schema in field_schema["oneOf"]:
            resolved_type = _resolve_json_type(sub_schema)
            types.append(resolved_type)
        if len(types) == 1:
            return types[0]
        from typing import Union

        return Union[tuple(types)]  # type: ignore # noqa: UP007

    # Handle allOf - for typing, we'll use the first type or Any
    # (proper allOf merging would require schema composition)
    if "allOf" in field_schema:
        for sub_schema in field_schema["allOf"]:
            if "type" in sub_schema:
                return _resolve_json_type(sub_schema)
        return Any  # type: ignore

    json_type = field_schema.get("type", "string")
    json_type = json_type[0] if isinstance(json_type, list) else json_type

    if json_type == "array":
        items_schema = field_schema.get("items", {})
        if items_schema:
            item_type = _resolve_json_type(items_schema)
            return list[item_type]  # type: ignore
        return list[Any]

    return json_type_mapping.get(json_type, str)


def _create_array_model(
    schema: dict[str, Any], model_name: str = "DynamicArrayModel"
) -> type[BaseModel]:
    """Create a model for root-level array schemas with proper item handling."""
    items_schema = schema.get("items", {})

    # If items are objects, create a nested model for proper field sanitization
    if items_schema.get("type") == "object" or "properties" in items_schema:
        item_model = create_model_from_json_schema(items_schema, f"{model_name}Item")
        list_type = list[item_model]  # type: ignore
    else:
        # For primitive types
        item_type = _resolve_json_type(items_schema) if items_schema else Any
        list_type = list[item_type]  # type: ignore

    class ArrayModel(BaseModel):
        items: list_type = Field(default_factory=list)  # type: ignore

        def model_dump(  # type: ignore
            self,
            *,
            mode: Literal["json", "python"] | str = "python",
            include: IncEx | None = None,
            exclude: IncEx | None = None,
            context: Any | None = None,
            by_alias: bool | None = True,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
            exclude_computed_fields: bool = False,
            round_trip: bool = False,
            warnings: bool | Literal["none", "warn", "error"] = True,
            fallback: Callable[[Any], Any] | None = None,
            serialize_as_any: bool = False,
        ) -> list[Any]:
            """Return the array directly, not wrapped in dict."""
            dumped_items = []
            for item in self.items:
                if isinstance(item, BaseModel):
                    dumped_items.append(
                        item.model_dump(  # type: ignore
                            mode=mode,
                            include=include,
                            exclude=exclude,
                            context=context,
                            by_alias=by_alias,
                            exclude_unset=exclude_unset,
                            exclude_defaults=exclude_defaults,
                            exclude_none=exclude_none,
                            round_trip=round_trip,
                            warnings=warnings,
                            serialize_as_any=serialize_as_any,
                        )
                    )
                else:
                    dumped_items.append(item)
            return dumped_items  # type: ignore

        def model_dump_json(
            self,
            *,
            indent: int | None = None,
            ensure_ascii: bool = False,
            include: IncEx | None = None,
            exclude: IncEx | None = None,
            context: Any | None = None,
            by_alias: bool | None = None,
            exclude_unset: bool = False,
            exclude_defaults: bool = False,
            exclude_none: bool = False,
            exclude_computed_fields: bool = False,
            round_trip: bool = False,
            warnings: bool | Literal["none", "warn", "error"] = True,
            fallback: Callable[[Any], Any] | None = None,
            serialize_as_any: bool = False,
            polymorphic_serialization: bool | None = None,
        ) -> str:
            """Return JSON array directly, not wrapped in object."""
            import json

            dumped_items = self.model_dump(
                mode="python",
                include=include,
                exclude=exclude,
                context=context,
                by_alias=by_alias,
                exclude_unset=exclude_unset,
                exclude_defaults=exclude_defaults,
                exclude_none=exclude_none,
                round_trip=round_trip,
                warnings=warnings,
                serialize_as_any=serialize_as_any,
            )
            return json.dumps(dumped_items, indent=indent, default=str)

        @classmethod
        def model_validate(
            cls,
            obj: Any,
            *,
            strict: bool | None = None,
            extra: ExtraValues | None = None,
            from_attributes: bool | None = None,
            context: Any | None = None,
            by_alias: bool | None = None,
            by_name: bool | None = None,
        ) -> Self:
            """Accept array data directly."""
            if isinstance(obj, list):
                return cls(items=obj)
            elif isinstance(obj, dict) and "items" in obj:
                return super().model_validate(
                    obj,
                    strict=strict,
                    from_attributes=from_attributes,
                    context=context,
                )
            else:
                raise ValueError(
                    f"Expected list or dict with 'items' key, got {type(obj)}"
                )

        @classmethod
        def model_json_schema(
            cls,
            by_alias: bool = True,
            ref_template: str = DEFAULT_REF_TEMPLATE,
            schema_generator: type[GenerateJsonSchema] = GenerateJsonSchema,
            mode: JsonSchemaMode = "validation",
            *,
            union_format: Literal["any_of", "primitive_type_array"] = "any_of",
        ) -> dict[str, Any]:
            """Return the original array schema, not wrapped in object schema."""
            return schema

        model_config = ConfigDict(populate_by_name=True, use_attribute_docstrings=True)

    ArrayModel.__name__ = model_name
    ArrayModel.__qualname__ = model_name
    return ArrayModel


def create_model_from_json_schema(
    schema: dict[str, Any], model_name: str = "DynamicModel"
) -> type[BaseModel]:
    """Create a Pydantic model from JSON Schema, handling underscore field names.

    Args:
        schema: A JSON Schema dictionary containing properties and required fields
        model_name: The name of the model

    Returns:
        A Pydantic model class with sanitized field names
    """
    # Initial validation of the schema
    _validate_json_schema(schema)

    # Handle array schemas at root level
    if schema.get("type") == "array":
        return _create_array_model(schema, model_name)

    # Original object handling logic (unchanged)
    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    field_aliases: dict[str, str] = {}

    used_field_names: set[str] = set()
    for original_field_name, field_schema in properties.items():
        sanitized_field_name = _sanitize_field_name(original_field_name)
        sanitized_field_name = _resolve_field_name_collisions(
            sanitized_field_name, used_field_names
        )
        used_field_names.add(sanitized_field_name)

        # Handle type mapping
        base_field_type = _resolve_json_type(field_schema)

        # Handle required vs optional
        # If the field already has None in its type (from anyOf with null), respect that
        has_null_in_union = (
            hasattr(base_field_type, "__args__")
            and type(None) in base_field_type.__args__
        )

        if original_field_name in required_fields:
            if has_null_in_union:
                # Field is required but allows null
                field_type = base_field_type
                default_value = field_schema.get("default", ...)
            else:
                # Field is required and doesn't allow null
                default_value = ...
                field_type = base_field_type
        else:
            # Field is optional
            if has_null_in_union:
                # Already has None in union
                field_type = base_field_type
            else:
                # Add None to make it optional
                field_type = base_field_type | None
            default_value = field_schema.get("default", None)

        # Create field with alias if name was sanitized
        field_kwargs = {"description": field_schema.get("description", "")}
        if sanitized_field_name != original_field_name:
            field_kwargs["alias"] = original_field_name
            field_aliases[sanitized_field_name] = original_field_name

        fields[sanitized_field_name] = (
            field_type,
            Field(default_value, **field_kwargs),
        )

    # Create the base dynamic model
    DynamicBaseModel: type[BaseModel] = create_model(model_name, **fields)

    class CustomModel(DynamicBaseModel):  # type: ignore
        def model_dump(self, by_alias: bool = True, **kwargs: Any) -> dict[str, Any]:
            obj: dict[str, Any] = super().model_dump(by_alias=by_alias, **kwargs)
            return obj

        def model_dump_json(self, by_alias: bool = True, **kwargs: Any) -> str:
            json: str = super().model_dump_json(by_alias=by_alias, **kwargs)
            return json

        @classmethod
        def model_json_schema(
            cls,
            by_alias: bool = True,
            ref_template: str = "#/$defs/{model}",
            schema_generator: Any = None,
            mode: str = "validation",
        ) -> dict[str, Any]:
            """Return the original schema, not Pydantic's generated schema."""
            return schema

        model_config = ConfigDict(populate_by_name=True, use_attribute_docstrings=True)

    CustomModel.__name__ = model_name
    CustomModel.__qualname__ = model_name

    return CustomModel
