import json
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from private_gpt.chat.schema_models import create_model_from_json_schema


def test_array_schema_handling() -> None:
    """Test handling of array schemas at root level."""
    # Test simple array of strings
    string_array_schema = {
        "type": "array",
        "items": {"type": "string"},
        "description": "Array of strings",
    }

    model = create_model_from_json_schema(string_array_schema, "StringArray")
    assert issubclass(model, BaseModel)

    # Test that model_json_schema returns the original schema
    schema = model.model_json_schema()
    assert schema == string_array_schema

    # Test validation with array data
    test_data = ["hello", "world", "test"]
    instance = model.model_validate(test_data)

    # Test that model_dump returns the array directly (not wrapped)
    dumped = instance.model_dump()
    assert dumped == test_data
    assert isinstance(dumped, list)

    # Test that model_dump_json returns JSON array directly
    import json

    json_str = instance.model_dump_json()
    json_data = json.loads(json_str)
    assert json_data == test_data
    assert isinstance(json_data, list)

    # Test array of objects with field sanitization
    object_array_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "_field": {"type": "string", "description": "Field with underscore"},
                "class": {"type": "string", "description": "Keyword field"},
                "normal": {"type": "number", "description": "Normal field"},
            },
            "required": ["_field"],
        },
    }

    object_model = create_model_from_json_schema(object_array_schema, "ObjectArray")
    assert issubclass(object_model, BaseModel)

    # Test that model_json_schema returns the original schema for object arrays
    object_schema = object_model.model_json_schema()
    assert object_schema == object_array_schema

    # Test with data that has field name issues
    test_object_data = [
        {"_field": "test1", "class": "MyClass", "normal": 1.5},
        {"_field": "test2", "class": "YourClass", "normal": 2.5},
    ]

    object_instance = object_model.model_validate(test_object_data)

    # The result should be the original array with proper field names preserved
    dumped_objects = object_instance.model_dump()
    assert isinstance(dumped_objects, list)
    assert len(dumped_objects) == 2
    assert dumped_objects[0]["_field"] == "test1"
    assert dumped_objects[0]["class"] == "MyClass"
    assert dumped_objects[1]["_field"] == "test2"

    # Test that nested objects have proper field sanitization
    # The internal model should have sanitized field
    # names but aliases preserve originals
    item_instance = object_instance.items[0]
    assert hasattr(
        item_instance, "field"
    )  # _field becomes field (leading underscore removed)
    assert hasattr(item_instance, "class_")  # class becomes class_ (keyword sanitized)

    # Test empty array
    empty_instance = object_model.model_validate([])
    empty_dumped = empty_instance.model_dump()
    assert empty_dumped == []
    assert isinstance(empty_dumped, list)

    # Test an object that contains an array of objects (1 level)
    object_array_schema = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"description": {"type": "string"}},
                    "required": ["description"],
                },
            }
        },
        "required": ["steps"],
    }
    object_array_model = create_model_from_json_schema(
        object_array_schema, "ObjectWithArray"
    )
    assert issubclass(object_array_model, BaseModel)
    # Test that model_json_schema returns the original schema for object with array
    object_array_schema_result = object_array_model.model_json_schema()
    assert object_array_schema_result == object_array_schema


@pytest.mark.parametrize(
    "schema_case",
    [
        # Case 1: Root-level array with complex nested objects
        {
            "name": "root_array_complex_nested",
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "user_id": {
                            "type": "integer",
                            "description": "User identifier",
                        },
                        "profile": {
                            "type": "object",
                            "properties": {
                                "personal_info": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "age": {"type": "integer", "minimum": 0},
                                        "contacts": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "type": {
                                                        "type": "string",
                                                        "enum": [
                                                            "email",
                                                            "phone",
                                                            "address",
                                                        ],
                                                    },
                                                    "value": {"type": "string"},
                                                    "is_primary": {
                                                        "type": "boolean",
                                                        "default": False,
                                                    },
                                                },
                                                "required": ["type", "value"],
                                            },
                                        },
                                    },
                                    "required": ["name", "age"],
                                },
                                "preferences": {
                                    "type": "object",
                                    "properties": {
                                        "notifications": {
                                            "type": "object",
                                            "properties": {
                                                "email": {
                                                    "type": "boolean",
                                                    "default": True,
                                                },
                                                "sms": {
                                                    "type": "boolean",
                                                    "default": False,
                                                },
                                                "push": {
                                                    "type": "boolean",
                                                    "default": True,
                                                },
                                            },
                                        },
                                        "privacy_settings": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "enum": [
                                                    "public",
                                                    "friends",
                                                    "private",
                                                ],
                                            },
                                        },
                                    },
                                },
                            },
                            "required": ["personal_info"],
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "created_at": {"type": "string", "format": "date-time"},
                                "tags": {"type": "array", "items": {"type": "string"}},
                                "scores": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                },
                                "config": {
                                    "type": "object",
                                    "additionalProperties": True,
                                },
                            },
                        },
                    },
                    "required": ["user_id", "profile"],
                },
                "minItems": 1,
            },
            "test_data": [
                {
                    "user_id": 12345,
                    "profile": {
                        "personal_info": {
                            "name": "John Doe",
                            "age": 30,
                            "contacts": [
                                {
                                    "type": "email",
                                    "value": "john@example.com",
                                    "is_primary": True,
                                },
                                {"type": "phone", "value": "+1234567890"},
                            ],
                        },
                        "preferences": {
                            "notifications": {
                                "email": True,
                                "sms": False,
                                "push": True,
                            },
                            "privacy_settings": ["friends", "private"],
                        },
                    },
                    "metadata": {
                        "created_at": "2024-01-15T10:30:00Z",
                        "tags": ["premium", "verified"],
                        "scores": [85.5, 92.0, 78.3],
                        "config": {"theme": "dark", "language": "en", "max_items": 100},
                    },
                },
                {
                    "user_id": 67890,
                    "profile": {
                        "personal_info": {
                            "name": "Jane Smith",
                            "age": 28,
                            "contacts": [
                                {"type": "email", "value": "jane@example.com"}
                            ],
                        }
                    },
                    "metadata": {"tags": [], "scores": [95.2], "config": {}},
                },
            ],
        },
        # Case 2: Complex object with multiple array types and field sanitization
        {
            "name": "complex_object_field_sanitization",
            "schema": {
                "type": "object",
                "properties": {
                    "_id": {"type": "string", "description": "Document identifier"},
                    "class": {"type": "string", "description": "Document class"},
                    "def": {"type": "string", "description": "Definition field"},
                    "data_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "timestamp": {"type": "integer"},
                                "value": {"type": "number"},
                                "_metadata": {
                                    "type": "object",
                                    "properties": {
                                        "source": {"type": "string"},
                                        "quality": {
                                            "type": "string",
                                            "enum": ["high", "medium", "low"],
                                        },
                                    },
                                },
                            },
                            "required": ["timestamp", "value"],
                        },
                    },
                    "matrix": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "number"}},
                    },
                    "nested_structure": {
                        "type": "object",
                        "properties": {
                            "level_1": {
                                "type": "object",
                                "properties": {
                                    "level_2": {
                                        "type": "object",
                                        "properties": {
                                            "level_3": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "from": {"type": "string"},
                                                        "to": {"type": "string"},
                                                        "weight": {"type": "number"},
                                                    },
                                                },
                                            }
                                        },
                                    }
                                },
                            }
                        },
                    },
                },
                "required": ["_id", "class", "data_points"],
            },
            "test_data": {
                "_id": "doc_123",
                "class": "sensor_data",
                "def": "Temperature sensor readings",
                "data_points": [
                    {
                        "timestamp": 1640995200,
                        "value": 23.5,
                        "_metadata": {"source": "sensor_01", "quality": "high"},
                    },
                    {
                        "timestamp": 1640995260,
                        "value": 24.1,
                        "_metadata": {"source": "sensor_01", "quality": "medium"},
                    },
                ],
                "matrix": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                "nested_structure": {
                    "level_1": {
                        "level_2": {
                            "level_3": [
                                {"from": "A", "to": "B", "weight": 0.8},
                                {"from": "B", "to": "C", "weight": 0.6},
                            ]
                        }
                    }
                },
            },
        },
        # Case 3: Deeply nested object with circular-like references and complex arrays
        {
            "name": "deeply_nested_circular_like",
            "schema": {
                "type": "object",
                "properties": {
                    "workflow": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step_id": {"type": "string"},
                                        "action": {"type": "string"},
                                        "parameters": {
                                            "type": "object",
                                            "additionalProperties": True,
                                        },
                                        "conditions": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "field": {"type": "string"},
                                                    "operator": {
                                                        "type": "string",
                                                        "enum": [
                                                            "eq",
                                                            "ne",
                                                            "gt",
                                                            "lt",
                                                            "in",
                                                        ],
                                                    },
                                                    "value": {},
                                                    "nested_conditions": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "field": {
                                                                    "type": "string"
                                                                },
                                                                "operator": {
                                                                    "type": "string",
                                                                    "enum": [
                                                                        "eq",
                                                                        "ne",
                                                                        "gt",
                                                                        "lt",
                                                                        "in",
                                                                    ],
                                                                },
                                                                "value": {},
                                                            },
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                        "outputs": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "name": {"type": "string"},
                                                    "type": {"type": "string"},
                                                    "transformations": {
                                                        "type": "array",
                                                        "items": {
                                                            "type": "object",
                                                            "properties": {
                                                                "function": {
                                                                    "type": "string"
                                                                },
                                                                "params": {
                                                                    "type": "array",
                                                                    "items": {},
                                                                },
                                                            },
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    }
                },
            },
            "test_data": {
                "workflow": {
                    "name": "data_processing_pipeline",
                    "steps": [
                        {
                            "step_id": "extract",
                            "action": "extract_data",
                            "parameters": {
                                "source": "database",
                                "query": "SELECT * FROM users",
                                "timeout": 30,
                            },
                            "conditions": [
                                {
                                    "field": "status",
                                    "operator": "eq",
                                    "value": "active",
                                    "nested_conditions": [
                                        {
                                            "field": "last_login",
                                            "operator": "gt",
                                            "value": "2024-01-01",
                                        }
                                    ],
                                }
                            ],
                            "outputs": [
                                {
                                    "name": "user_data",
                                    "type": "dataframe",
                                    "transformations": [
                                        {
                                            "function": "clean_nulls",
                                            "params": ["name", "email"],
                                        },
                                        {"function": "validate_email", "params": []},
                                    ],
                                }
                            ],
                        },
                        {
                            "step_id": "transform",
                            "action": "transform_data",
                            "parameters": {
                                "transformations": ["normalize", "encode"],
                                "encoding": "utf-8",
                            },
                            "conditions": [],
                            "outputs": [
                                {
                                    "name": "processed_data",
                                    "type": "array",
                                    "transformations": [],
                                }
                            ],
                        },
                    ],
                }
            },
        },
        # Case 4: Tuple-style array items (items as array instead of object)
        {
            "name": "tuple_style_array_items",
            "schema": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": {"type": "string"},
                                "_id": {"type": "string"},
                                "class": {"type": "string"},
                            },
                            "required": ["description"],
                        },
                    }
                },
                "required": ["steps"],
            },
            "test_data": {
                "steps": [
                    {
                        "description": "First step description",
                        "_id": "step_1",
                        "class": "ProcessStep",
                    }
                ]
            },
        },
        # Case 5: CLI-style tool schema with
        # dash-prefixed fields and additionalProperties
        {
            "name": "cli_tool_dash_prefixed_fields",
            "schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in",
                    },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                        "description": "Output mode",
                    },
                    "-B": {
                        "type": "number",
                        "description": "Lines before match",
                    },
                    "-A": {
                        "type": "number",
                        "description": "Lines after match",
                    },
                    "-C": {
                        "type": "number",
                        "description": "Lines before and after match",
                    },
                    "-n": {
                        "type": "boolean",
                        "description": "Show line numbers",
                    },
                    "-i": {
                        "type": "boolean",
                        "description": "Case insensitive",
                    },
                    "head_limit": {
                        "type": "number",
                        "description": "Limit output to first N lines",
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
            "test_data": {
                "pattern": "log.*Error",
                "path": "/var/log",
                "output_mode": "content",
                "-B": 2,
                "-A": 3,
                "-n": True,
                "-i": False,
                "head_limit": 100,
            },
        },
        # Case 6: Tool schema with dash-prefixed fields and anyOf types
        {
            "name": "tool_schema_dash_fields_anyof",
            "schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to execute",
                    },
                    "-v": {
                        "anyOf": [{"type": "boolean"}, {"type": "null"}],
                        "default": None,
                        "description": "Verbose output",
                    },
                    "-o": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": "Output file path",
                    },
                    "-n": {
                        "anyOf": [{"type": "number"}, {"type": "null"}],
                        "default": None,
                        "description": "Number of results",
                    },
                    "format": {
                        "type": "string",
                        "enum": ["json", "yaml", "text", "csv"],
                        "description": "Output format",
                    },
                    "filters": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"},
                        ],
                        "default": None,
                        "description": "List of filter expressions",
                    },
                },
                "required": ["command", "format"],
                "additionalProperties": False,
            },
            "test_data": {
                "command": "list-resources",
                "format": "json",
                "-v": True,
                "-o": "/tmp/output.json",
                "-n": 50,
                "filters": ["status=active", "region=us-east-1"],
            },
        },
        # Case 7: Flat tool schema mixing enums, dash-flags,
        # and additionalProperties: false
        {
            "name": "flat_tool_enum_and_dash_flags",
            "schema": {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string",
                    },
                    "index": {
                        "type": "string",
                        "description": "Index name to search",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["exact", "fuzzy", "semantic", "hybrid"],
                        "description": "Search type",
                    },
                    "size": {
                        "type": "number",
                        "description": "Number of results to return",
                    },
                    "from": {
                        "type": "number",
                        "description": "Offset for pagination",
                    },
                    "-s": {
                        "type": "boolean",
                        "description": "Silent mode, suppress warnings",
                    },
                    "-p": {
                        "type": "boolean",
                        "description": "Pretty-print output",
                    },
                    "-r": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                        "description": "Remote endpoint override",
                    },
                    "fields": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"},
                        ],
                        "default": None,
                        "description": "Fields to include in response",
                    },
                    "sort": {
                        "anyOf": [
                            {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "field": {"type": "string"},
                                        "order": {
                                            "type": "string",
                                            "enum": ["asc", "desc"],
                                        },
                                    },
                                    "required": ["field", "order"],
                                },
                            },
                            {"type": "null"},
                        ],
                        "default": None,
                        "description": "Sort criteria",
                    },
                },
                "required": ["query", "index", "type"],
                "additionalProperties": False,
            },
            "test_data": {
                "query": "error rate spike",
                "index": "logs-2024",
                "type": "hybrid",
                "size": 25,
                "from": 0,
                "-s": False,
                "-p": True,
                "-r": None,
                "fields": ["timestamp", "message", "level"],
                "sort": [{"field": "timestamp", "order": "desc"}],
            },
        },
    ],
)
def test_complex_json_schema_handling(schema_case: dict[str, Any]) -> None:
    """Test handling of complex JSON schemas with various nested structures."""
    schema = schema_case["schema"]
    test_data = schema_case["test_data"]
    case_name = schema_case["name"]

    # Create model from schema
    model = create_model_from_json_schema(schema, f"ComplexModel_{case_name}")
    assert issubclass(model, BaseModel)

    # Test that model_json_schema returns the original schema
    generated_schema = model.model_json_schema()
    assert generated_schema == schema, f"Schema mismatch for case: {case_name}"

    # Test validation with test data
    instance = model.model_validate(test_data)
    assert instance is not None

    # Test model_dump returns data in expected format
    dumped = instance.model_dump()

    if schema["type"] == "array":
        # For root-level arrays, dumped should be a list
        assert isinstance(
            dumped, list
        ), f"Expected list for array schema in case: {case_name}"
        assert dumped == test_data, f"Data mismatch for case: {case_name}"
    else:
        # For objects, dumped should be a dict
        assert isinstance(
            dumped, dict
        ), f"Expected dict for object schema in case: {case_name}"

        # Verify all required fields are present and correctly mapped
        _verify_data_integrity(dumped, test_data, case_name)

    # Test JSON serialization/deserialization
    json_str = instance.model_dump_json()
    json_data = json.loads(json_str)

    if schema["type"] == "array":
        assert isinstance(
            json_data, list
        ), f"JSON should be array for case: {case_name}"
    else:
        assert isinstance(
            json_data, dict
        ), f"JSON should be object for case: {case_name}"

    # Test that we can create a new instance from dumped data
    new_instance = model.model_validate(dumped)
    assert (
        new_instance.model_dump() == dumped
    ), f"Round-trip validation failed for case: {case_name}"

    # Test field sanitization for object schemas
    if schema["type"] == "object":
        _verify_field_sanitization(instance, schema, case_name)


def _verify_data_integrity(
    dumped: dict[str, Any], original: dict[str, Any], case_name: str
) -> None:
    """Verify that dumped data maintains integrity with original data."""
    # Check that all original keys are preserved in dumped data
    for key, value in original.items():
        assert (
            key in dumped
        ), f"Key '{key}' missing in dumped data for case: {case_name}"

        if isinstance(value, dict):
            assert isinstance(
                dumped[key], dict
            ), f"Value type mismatch for key '{key}' in case: {case_name}"
            _verify_data_integrity(dumped[key], value, case_name)
        elif isinstance(value, list):
            assert isinstance(
                dumped[key], list
            ), f"Value type mismatch for key '{key}' in case: {case_name}"
            assert len(dumped[key]) == len(
                value
            ), f"Array length mismatch for key '{key}' in case: {case_name}"
        else:
            assert (
                dumped[key] == value
            ), f"Value mismatch for key '{key}' in case: {case_name}"


def _verify_field_sanitization(
    instance: BaseModel, schema: dict[str, Any], case_name: str
) -> None:
    """Verify that field names are properly sanitized while maintaining aliases."""
    if "properties" in schema:
        for field_name in schema["properties"]:
            # Check for fields that should be sanitized
            if field_name.startswith("_"):
                sanitized_name = field_name.lstrip("_")
                if sanitized_name:  # Only if there's something left after stripping
                    assert hasattr(
                        instance, sanitized_name
                    ), f"Sanitized field '{sanitized_name}' not found for case: {case_name}"

            elif field_name in [
                "class",
                "def",
                "from",
                "to",
                "if",
                "else",
                "for",
                "while",
                "return",
                "import",
            ]:
                # Python keywords should be sanitized
                sanitized_name = f"{field_name}_"
                assert hasattr(
                    instance, sanitized_name
                ), f"Sanitized keyword field '{sanitized_name}' not found for case: {case_name}"


def test_tuple_style_array_items_raises_error() -> None:
    """Test that tuple-style array items (items as array) raise ValueError."""
    # Schema with tuple-style items (items is an array) - should crash
    tuple_array_schema = {
        "type": "array",
        "items": [
            {
                "type": "object",
                "properties": {"description": {"type": "string"}},
                "required": ["description"],
            }
        ],
    }

    # This should raise ValueError
    with pytest.raises(ValueError):
        create_model_from_json_schema(tuple_array_schema, "TupleArrayModel")


def test_nested_tuple_style_array_items_raises_error() -> None:
    """Test that nested tuple-style array items also raise ValueError."""
    # Object schema with nested array that has tuple-style items
    nested_tuple_schema = {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": [
                    {
                        "type": "object",
                        "properties": {"description": {"type": "string"}},
                        "required": ["description"],
                    }
                ],
            }
        },
        "required": ["steps"],
    }

    # This should also raise ValueError when processing the nested array
    with pytest.raises(ValueError):
        create_model_from_json_schema(nested_tuple_schema, "NestedTupleModel")


def test_object_schema_backward_compatibility() -> None:
    """Test that object schemas still work as before."""
    object_schema = {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "First parameter"},
            "_param2": {
                "type": "number",
                "description": "Second parameter with underscore",
            },
            "normal_param": {"type": "boolean", "description": "Normal parameter"},
            "class": {"type": "string", "description": "Keyword field"},
            "match": {"type": "string", "description": "Soft keyword field"},
        },
        "required": ["param1"],
    }

    model = create_model_from_json_schema(object_schema, "TestModel")
    assert issubclass(model, BaseModel)

    # Create instance with original field names (using aliases)
    instance = model(
        param1="test_value",
        param2=42.5,  # sanitized field name
        normal_param=True,
        class_="test_class",  # keyword sanitized
        match_="test_match",  # soft keyword sanitized
    )

    # Test default serialization returns original field names
    serialized = instance.model_dump()

    # Should contain original field names with underscores by default
    assert "param1" in serialized
    assert "_param2" in serialized
    assert "normal_param" in serialized
    assert "class" in serialized
    assert "match" in serialized

    # Values should be preserved
    assert serialized["param1"] == "test_value"
    assert serialized["_param2"] == 42.5
    assert serialized["normal_param"] is True
    assert serialized["class"] == "test_class"
    assert serialized["match"] == "test_match"

    # Test JSON serialization also uses original names by default
    import json

    json_str = instance.model_dump_json()
    json_data = json.loads(json_str)

    assert "_param2" in json_data
    assert json_data["_param2"] == 42.5
    assert "class" in json_data
    assert "match" in json_data


def test_anyof_with_null_optional_field() -> None:
    """Test handling of anyOf with null type (your specific case)."""
    schema = {
        "properties": {
            "a": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "default": None,
                "description": "The first number to multiply",
                "title": "A",
            },
            "b": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "default": None,
                "description": "The second number to multiply",
                "title": "B",
            },
        },
        "title": "multiply_Schema",
        "type": "object",
    }

    model = create_model_from_json_schema(schema, "MultiplySchema")
    assert issubclass(model, BaseModel)

    # Test with both values provided
    instance1 = model.model_validate({"a": 5.0, "b": 10.0})
    assert instance1.a == 5.0
    assert instance1.b == 10.0

    # Test with null values
    instance2 = model.model_validate({"a": None, "b": None})
    assert instance2.a is None
    assert instance2.b is None

    # Test with mixed values
    instance3 = model.model_validate({"a": 3.14, "b": None})
    assert instance3.a == 3.14
    assert instance3.b is None

    # Test with missing values (should use defaults)
    instance4 = model.model_validate({})
    assert instance4.a is None
    assert instance4.b is None

    # Test serialization preserves original field names
    dumped = instance1.model_dump()
    assert dumped == {"a": 5.0, "b": 10.0}


def test_anyof_multiple_types() -> None:
    """Test anyOf with multiple non-null types."""
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}],
                "description": "Can be string, number, or boolean",
            }
        },
        "required": ["value"],
    }

    model = create_model_from_json_schema(schema, "AnyOfModel")

    # Test with string
    instance1 = model.model_validate({"value": "test"})
    assert instance1.value == "test"

    # Test with number
    instance2 = model.model_validate({"value": 42})
    assert instance2.value == 42

    # Test with boolean
    instance3 = model.model_validate({"value": True})
    assert instance3.value is True


def test_anyof_with_array() -> None:
    """Test anyOf containing array type."""
    schema = {
        "type": "object",
        "properties": {
            "data": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "null"},
                ],
                "description": "Optional array of strings",
            }
        },
    }

    model = create_model_from_json_schema(schema, "AnyOfArrayModel")

    # Test with array
    instance1 = model.model_validate({"data": ["a", "b", "c"]})
    assert instance1.data == ["a", "b", "c"]

    # Test with null
    instance2 = model.model_validate({"data": None})
    assert instance2.data is None

    # Test with missing (should be None)
    instance3 = model.model_validate({})
    assert instance3.data is None


def test_oneof_handling() -> None:
    """Test oneOf schema handling."""
    schema = {
        "type": "object",
        "properties": {
            "id": {
                "oneOf": [{"type": "string"}, {"type": "integer"}],
                "description": "ID can be string or integer",
            }
        },
        "required": ["id"],
    }

    model = create_model_from_json_schema(schema, "OneOfModel")

    # Test with string
    instance1 = model.model_validate({"id": "abc123"})
    assert instance1.id == "abc123"

    # Test with integer
    instance2 = model.model_validate({"id": 123})
    assert instance2.id == 123


def test_allof_handling() -> None:
    """Test allOf schema handling - takes first type."""
    schema = {
        "type": "object",
        "properties": {
            "name": {
                "allOf": [
                    {"type": "string", "minLength": 1},
                    {"type": "string", "maxLength": 100},
                ],
                "description": "Name with constraints",
            }
        },
        "required": ["name"],
    }

    model = create_model_from_json_schema(schema, "AllOfModel")

    # Should accept string
    instance = model.model_validate({"name": "John Doe"})
    assert instance.name == "John Doe"


def test_nested_anyof_in_array() -> None:
    """Test anyOf inside array items."""
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "anyOf": [{"type": "string"}, {"type": "number"}],
                },
            }
        },
    }

    model = create_model_from_json_schema(schema, "NestedAnyOfModel")

    # Test with mixed types in array
    instance = model.model_validate({"items": ["test", 42, "another", 3.14]})
    assert instance.items == ["test", 42, "another", 3.14]


def test_anyof_with_object_types() -> None:
    """Test anyOf with different object schemas."""
    schema = {
        "type": "object",
        "properties": {
            "response": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {"success": {"type": "boolean"}},
                    },
                    {
                        "type": "object",
                        "properties": {"error": {"type": "string"}},
                    },
                ],
            }
        },
    }

    model = create_model_from_json_schema(schema, "AnyOfObjectModel")

    # Test with first schema variant
    instance1 = model.model_validate({"response": {"success": True}})
    assert instance1.response == {"success": True}

    # Test with second schema variant
    instance2 = model.model_validate({"response": {"error": "Failed"}})
    assert instance2.response == {"error": "Failed"}


def test_complex_anyof_with_required_and_optional() -> None:
    """Test complex schema with anyOf in both required and optional fields."""
    schema = {
        "type": "object",
        "properties": {
            "required_field": {
                "anyOf": [{"type": "string"}, {"type": "integer"}],
                "description": "Required field with union type",
            },
            "optional_field": {
                "anyOf": [{"type": "boolean"}, {"type": "null"}],
                "default": None,
                "description": "Optional field with null",
            },
            "another_optional": {
                "anyOf": [
                    {"type": "array", "items": {"type": "number"}},
                    {"type": "null"},
                ],
                "description": "Optional array",
            },
        },
        "required": ["required_field"],
    }

    model = create_model_from_json_schema(schema, "ComplexAnyOfModel")

    # Test with all fields
    instance1 = model.model_validate(
        {
            "required_field": "test",
            "optional_field": True,
            "another_optional": [1.5, 2.5],
        }
    )
    assert instance1.required_field == "test"
    assert instance1.optional_field is True
    assert instance1.another_optional == [1.5, 2.5]

    # Test with only required field
    instance2 = model.model_validate({"required_field": 42})
    assert instance2.required_field == 42
    assert instance2.optional_field is None
    assert instance2.another_optional is None

    # Test with null optional fields
    instance3 = model.model_validate(
        {"required_field": "value", "optional_field": None, "another_optional": None}
    )
    assert instance3.required_field == "value"
    assert instance3.optional_field is None
    assert instance3.another_optional is None


def test_anyof_with_field_sanitization() -> None:
    """Test anyOf works correctly with field name sanitization."""
    schema = {
        "type": "object",
        "properties": {
            "_field": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "description": "Field with underscore",
            },
            "class": {
                "anyOf": [{"type": "integer"}, {"type": "null"}],
                "description": "Python keyword field",
            },
        },
        "required": ["class"],
    }

    model = create_model_from_json_schema(schema, "SanitizedAnyOfModel")

    # Test that fields are accessible with sanitized names
    instance = model.model_validate({"_field": "test", "class": 42})
    assert hasattr(instance, "field")  # _field becomes field
    assert hasattr(instance, "class_")  # class becomes class_

    # Test serialization uses original names
    dumped = instance.model_dump()
    assert "_field" in dumped
    assert "class" in dumped
    assert dumped["_field"] == "test"
    assert dumped["class"] == 42


def test_root_level_array_with_anyof_items() -> None:
    """Test root-level array where items use anyOf."""
    schema = {
        "type": "array",
        "items": {
            "anyOf": [{"type": "string"}, {"type": "number"}, {"type": "null"}],
        },
    }

    model = create_model_from_json_schema(schema, "ArrayAnyOfModel")

    # Test with mixed types
    test_data = ["hello", 42, None, "world", 3.14]
    instance = model.model_validate(test_data)

    dumped = instance.model_dump()
    assert dumped == test_data


def test_nested_anyof_in_object_in_array() -> None:
    """Test complex nesting: array of objects with anyOf fields."""
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": {
                    "anyOf": [{"type": "string"}, {"type": "integer"}],
                },
                "value": {
                    "anyOf": [{"type": "number"}, {"type": "null"}],
                    "default": None,
                },
            },
            "required": ["id"],
        },
    }

    model = create_model_from_json_schema(schema, "ComplexNestedAnyOfModel")

    test_data = [
        {"id": "abc", "value": 10.5},
        {"id": 123, "value": None},
        {"id": "xyz", "value": 42.0},
    ]

    instance = model.model_validate(test_data)
    dumped = instance.model_dump()

    assert len(dumped) == 3
    assert dumped[0]["id"] == "abc"
    assert dumped[0]["value"] == 10.5
    assert dumped[1]["id"] == 123
    assert dumped[1]["value"] is None
    assert dumped[2]["id"] == "xyz"
    assert dumped[2]["value"] == 42.0


def test_oneof_with_null() -> None:
    """Test oneOf including null type."""
    schema = {
        "type": "object",
        "properties": {
            "status": {
                "oneOf": [
                    {"type": "string", "enum": ["active", "inactive"]},
                    {"type": "null"},
                ],
            }
        },
    }

    model = create_model_from_json_schema(schema, "OneOfNullModel")

    # Test with string
    instance1 = model.model_validate({"status": "active"})
    assert instance1.status == "active"

    # Test with null
    instance2 = model.model_validate({"status": None})
    assert instance2.status is None


def test_allof_with_multiple_schemas() -> None:
    """Test allOf with multiple schemas - uses first with type."""
    schema = {
        "type": "object",
        "properties": {
            "email": {
                "allOf": [
                    {"type": "string"},
                    {"format": "email"},
                    {"minLength": 5},
                ],
            }
        },
        "required": ["email"],
    }

    model = create_model_from_json_schema(schema, "AllOfEmailModel")

    instance = model.model_validate({"email": "test@example.com"})
    assert instance.email == "test@example.com"


def test_mixed_combinators() -> None:
    """Test schema with multiple combinator types (anyOf, allOf, oneOf)."""
    schema = {
        "type": "object",
        "properties": {
            "field1": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
            },
            "field2": {
                "oneOf": [{"type": "integer"}, {"type": "boolean"}],
            },
            "field3": {
                "allOf": [{"type": "string"}, {"minLength": 1}],
            },
        },
        "required": ["field2", "field3"],
    }

    model = create_model_from_json_schema(schema, "MixedCombinatorsModel")

    instance = model.model_validate({"field1": "test", "field2": 42, "field3": "hello"})
    assert instance.field1 == "test"
    assert instance.field2 == 42
    assert instance.field3 == "hello"

    # Test with null for field1
    instance2 = model.model_validate({"field1": None, "field2": True, "field3": "hi"})
    assert instance2.field1 is None
    assert instance2.field2 is True
    assert instance2.field3 == "hi"


def test_anyof_preserves_schema() -> None:
    """Test that model_json_schema returns original schema with anyOf."""
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "default": None,
            }
        },
    }

    model = create_model_from_json_schema(schema, "PreserveSchemaModel")
    returned_schema = model.model_json_schema()

    assert returned_schema == schema


def test_deeply_nested_anyof() -> None:
    """Test anyOf deeply nested in complex structure."""
    schema = {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "properties": {
                    "nested": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "value": {
                                    "anyOf": [
                                        {"type": "string"},
                                        {"type": "number"},
                                        {"type": "null"},
                                    ],
                                }
                            },
                        },
                    }
                },
            }
        },
    }

    model = create_model_from_json_schema(schema, "DeeplyNestedAnyOfModel")

    test_data = {
        "data": {
            "nested": [
                {"value": "string"},
                {"value": 42},
                {"value": None},
                {"value": 3.14},
            ]
        }
    }

    instance = model.model_validate(test_data)
    dumped = instance.model_dump()

    assert dumped["data"]["nested"][0]["value"] == "string"
    assert dumped["data"]["nested"][1]["value"] == 42
    assert dumped["data"]["nested"][2]["value"] is None
    assert dumped["data"]["nested"][3]["value"] == 3.14


def test_json_serialization_with_anyof() -> None:
    """Test JSON serialization/deserialization with anyOf fields."""
    schema = {
        "type": "object",
        "properties": {
            "id": {
                "anyOf": [{"type": "string"}, {"type": "integer"}],
            },
            "optional": {
                "anyOf": [{"type": "number"}, {"type": "null"}],
                "default": None,
            },
        },
        "required": ["id"],
    }

    model = create_model_from_json_schema(schema, "JsonSerializationModel")

    # Create instance and serialize to JSON
    instance = model.model_validate({"id": "abc123", "optional": 42.5})
    json_str = instance.model_dump_json()
    json_data = json.loads(json_str)

    assert json_data["id"] == "abc123"
    assert json_data["optional"] == 42.5

    # Test round-trip
    new_instance = model.model_validate(json_data)
    assert new_instance.id == "abc123"
    assert new_instance.optional == 42.5


class TestComplexNestedSchemas:
    """Test deeply nested and complex object structures."""

    def test_deeply_nested_objects(self) -> None:
        """Test schema with multiple levels of nested objects."""
        schema = {
            "type": "object",
            "properties": {
                "level1": {
                    "type": "object",
                    "properties": {
                        "level2": {
                            "type": "object",
                            "properties": {
                                "level3": {
                                    "type": "object",
                                    "properties": {"deep_value": {"type": "string"}},
                                    "required": ["deep_value"],
                                }
                            },
                            "required": ["level3"],
                        }
                    },
                    "required": ["level2"],
                }
            },
            "required": ["level1"],
        }

        Model = create_model_from_json_schema(schema, "DeeplyNestedModel")

        # Valid nested data
        data = {"level1": {"level2": {"level3": {"deep_value": "found it!"}}}}

        instance = Model.model_validate(data)
        assert instance.level1["level2"]["level3"]["deep_value"] == "found it!"

    def test_mixed_nested_arrays_and_objects(self) -> None:
        """Test schema mixing nested arrays and objects."""
        schema = {
            "type": "object",
            "properties": {
                "users": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "addresses": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "street": {"type": "string"},
                                        "city": {"type": "string"},
                                        "coordinates": {
                                            "type": "array",
                                            "items": {"type": "number"},
                                        },
                                    },
                                    "required": ["street", "city"],
                                },
                            },
                        },
                        "required": ["name"],
                    },
                }
            },
            "required": ["users"],
        }

        Model = create_model_from_json_schema(schema, "MixedNestedModel")

        data = {
            "users": [
                {
                    "name": "Alice",
                    "addresses": [
                        {
                            "street": "123 Main St",
                            "city": "NYC",
                            "coordinates": [40.7128, -74.0060],
                        }
                    ],
                },
                {"name": "Bob", "addresses": []},
            ]
        }

        instance = Model.model_validate(data)
        assert len(instance.users) == 2
        assert instance.users[0]["name"] == "Alice"
        assert instance.users[0]["addresses"][0]["coordinates"] == [40.7128, -74.0060]


class TestComplexUnionTypes:
    """Test complex union type scenarios with anyOf, oneOf, allOf."""

    def test_anyof_with_multiple_complex_types(self) -> None:
        """Test anyOf with multiple object types."""
        schema = {
            "type": "object",
            "properties": {
                "data": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "value": {"type": "string"},
                            },
                            "required": ["type", "value"],
                        },
                        {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "count": {"type": "integer"},
                            },
                            "required": ["type", "count"],
                        },
                        {"type": "array", "items": {"type": "string"}},
                        {"type": "null"},
                    ]
                }
            },
            "required": [],
        }

        Model = create_model_from_json_schema(schema, "AnyOfComplexModel")

        # Test with first type
        instance1 = Model.model_validate({"data": {"type": "text", "value": "hello"}})
        assert instance1.data["type"] == "text"

        # Test with array type
        instance2 = Model.model_validate({"data": ["a", "b", "c"]})
        assert instance2.data == ["a", "b", "c"]

        # Test with null
        instance3 = Model.model_validate({"data": None})
        assert instance3.data is None

    def test_oneof_with_discriminator_like_pattern(self) -> None:
        """Test oneOf simulating discriminator pattern."""
        schema = {
            "type": "object",
            "properties": {
                "shape": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string"},
                                "radius": {"type": "number"},
                            },
                            "required": ["kind", "radius"],
                        },
                        {
                            "type": "object",
                            "properties": {
                                "kind": {"type": "string"},
                                "width": {"type": "number"},
                                "height": {"type": "number"},
                            },
                            "required": ["kind", "width", "height"],
                        },
                    ]
                }
            },
            "required": ["shape"],
        }

        Model = create_model_from_json_schema(schema, "OneOfDiscriminatorModel")

        circle = Model.model_validate({"shape": {"kind": "circle", "radius": 5.0}})
        assert circle.shape["radius"] == 5.0

        rectangle = Model.model_validate(
            {"shape": {"kind": "rectangle", "width": 10, "height": 20}}
        )
        assert rectangle.shape["width"] == 10

    def test_allof_composition(self) -> None:
        """Test allOf schema composition."""
        schema = {
            "type": "object",
            "properties": {
                "entity": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "created_at": {"type": "string"},
                            },
                        },
                        {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "email": {"type": "string"},
                            },
                        },
                    ]
                }
            },
            "required": [],
        }

        Model = create_model_from_json_schema(schema, "AllOfCompositionModel")

        data = {
            "entity": {
                "id": "123",
                "created_at": "2025-01-01",
                "name": "John",
                "email": "john@example.com",
            }
        }

        instance = Model.model_validate(data)
        assert instance.entity["id"] == "123"
        assert instance.entity["name"] == "John"

    def test_nested_anyof_in_array(self) -> None:
        """Test anyOf within array items."""
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "number"},
                            {
                                "type": "object",
                                "properties": {"nested": {"type": "boolean"}},
                            },
                        ]
                    },
                }
            },
            "required": [],
        }

        Model = create_model_from_json_schema(schema, "NestedAnyOfArrayModel")

        data = {"items": ["string_value", 42, 3.14, {"nested": True}, "another_string"]}

        instance = Model.model_validate(data)
        assert len(instance.items) == 5
        assert instance.items[0] == "string_value"
        assert instance.items[1] == 42
        assert instance.items[3]["nested"] is True


class TestReservedNamesAndSanitization:
    """Test handling of reserved names and field sanitization."""

    def test_all_pydantic_reserved_names(self) -> None:
        """Test all Pydantic reserved names are properly sanitized."""
        reserved_properties = {
            "model_config": {"type": "string"},
            "model_fields": {"type": "string"},
            "model_computed_fields": {"type": "string"},
            "model_dump": {"type": "string"},
            "model_validate": {"type": "string"},
            "dict": {"type": "string"},
            "json": {"type": "string"},
            "copy": {"type": "string"},
            "schema": {"type": "string"},
        }

        schema = {"type": "object", "properties": reserved_properties, "required": []}

        Model = create_model_from_json_schema(schema, "ReservedNamesModel")

        data = {name: f"value_{name}" for name in reserved_properties}
        instance = Model.model_validate(data)

        # Verify all fields are accessible and have correct values
        dumped = instance.model_dump(by_alias=True)
        for name in reserved_properties:
            assert dumped[name] == f"value_{name}"

    def test_python_keywords_as_fields(self) -> None:
        """Test Python keywords are properly handled."""
        schema = {
            "type": "object",
            "properties": {
                "class": {"type": "string"},
                "def": {"type": "string"},
                "return": {"type": "string"},
                "if": {"type": "string"},
                "else": {"type": "string"},
                "import": {"type": "string"},
                "from": {"type": "string"},
                "as": {"type": "string"},
                "try": {"type": "string"},
                "except": {"type": "string"},
            },
            "required": ["class", "def"],
        }

        Model = create_model_from_json_schema(schema, "KeywordsModel")

        data = {
            "class": "MyClass",
            "def": "my_function",
            "return": "value",
            "if": "condition",
        }

        instance = Model.model_validate(data)
        dumped = instance.model_dump(by_alias=True)
        assert dumped["class"] == "MyClass"
        assert dumped["def"] == "my_function"

    def test_invalid_identifier_characters(self) -> None:
        """Test fields with invalid Python identifier characters."""
        schema = {
            "type": "object",
            "properties": {
                "field-with-dashes": {"type": "string"},
                "field.with.dots": {"type": "string"},
                "field with spaces": {"type": "string"},
                "field@with#special$chars": {"type": "string"},
                "123numeric_start": {"type": "string"},
                "field/slash": {"type": "string"},
            },
            "required": [],
        }

        Model = create_model_from_json_schema(schema, "InvalidCharsModel")

        data = {
            "field-with-dashes": "value1",
            "field.with.dots": "value2",
            "field with spaces": "value3",
            "field@with#special$chars": "value4",
            "123numeric_start": "value5",
            "field/slash": "value6",
        }

        instance = Model.model_validate(data)
        dumped = instance.model_dump(by_alias=True)

        assert dumped["field-with-dashes"] == "value1"
        assert dumped["field.with.dots"] == "value2"
        assert dumped["field with spaces"] == "value3"
        assert dumped["123numeric_start"] == "value5"

    def test_dunder_names(self) -> None:
        """Test double underscore (dunder) names."""
        schema = {
            "type": "object",
            "properties": {
                "__init__": {"type": "string"},
                "__name__": {"type": "string"},
                "__dict__": {"type": "string"},
            },
            "required": [],
        }

        Model = create_model_from_json_schema(schema, "DunderNamesModel")

        data = {
            "__init__": "value1",
            "__name__": "value2",
            "__dict__": "value3",
        }

        instance = Model.model_validate(data)
        dumped = instance.model_dump(by_alias=True)
        assert dumped["__init__"] == "value1"

    def test_collision_resolution(self) -> None:
        """Test field name collision resolution."""
        schema = {
            "type": "object",
            "properties": {
                "field": {"type": "string"},
                "field_": {"type": "string"},
                "class": {"type": "string"},  # becomes class_
                "class_": {"type": "string"},  # collision with sanitized class
            },
            "required": [],
        }

        Model = create_model_from_json_schema(schema, "CollisionModel")

        data = {
            "field": "value1",
            "field_": "value2",
            "class": "value3",
            "class_": "value4",
        }

        instance = Model.model_validate(data)
        dumped = instance.model_dump(by_alias=True)

        assert dumped["field"] == "value1"
        assert dumped["field_"] == "value2"
        assert dumped["class"] == "value3"
        assert dumped["class_"] == "value4"


class TestComplexArraySchemas:
    """Test complex array schema scenarios."""

    def test_root_level_array_with_nested_objects(self) -> None:
        """Test array schema at root with complex nested objects."""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "attributes": {
                                "type": "object",
                                "properties": {
                                    "key": {"type": "string"},
                                    "value": {"type": "string"},
                                },
                            },
                        },
                    },
                },
                "required": ["id"],
            },
        }

        Model = create_model_from_json_schema(schema, "RootArrayNestedModel")

        data = [
            {
                "id": 1,
                "metadata": {
                    "tags": ["tag1", "tag2"],
                    "attributes": {"key": "color", "value": "blue"},
                },
            },
            {
                "id": 2,
                "metadata": {
                    "tags": [],
                    "attributes": {"key": "size", "value": "large"},
                },
            },
        ]

        instance = Model.model_validate(data)
        dumped = instance.model_dump()

        assert len(dumped) == 2
        assert dumped[0]["id"] == 1
        assert dumped[0]["metadata"]["tags"] == ["tag1", "tag2"]
        assert dumped[1]["metadata"]["attributes"]["value"] == "large"

    def test_root_array_json_serialization(self) -> None:
        """Test that root-level arrays serialize correctly to JSON."""
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "value": {"type": "number"}},
                "required": ["name", "value"],
            },
        }

        Model = create_model_from_json_schema(schema, "RootArrayJsonModel")

        data = [{"name": "item1", "value": 10.5}, {"name": "item2", "value": 20.3}]

        instance = Model.model_validate(data)
        json_str = instance.model_dump_json()
        parsed = json.loads(json_str)

        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "item1"

    def test_array_of_arrays(self) -> None:
        """Test nested array structures."""
        schema = {
            "type": "object",
            "properties": {
                "matrix": {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                }
            },
            "required": ["matrix"],
        }

        Model = create_model_from_json_schema(schema, "ArrayOfArraysModel")

        data = {"matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}

        instance = Model.model_validate(data)
        assert instance.matrix[1][1] == 5

    def test_array_with_anyof_items(self) -> None:
        """Test arrays where items can be different types."""
        schema = {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "data": {"anyOf": [{"type": "string"}, {"type": "number"}]},
                        },
                    },
                ]
            },
        }

        Model = create_model_from_json_schema(schema, "HeterogeneousArrayModel")

        data = [
            "string",
            42,
            3.14,
            {"type": "custom", "data": "value"},
            {"type": "other", "data": 100},
        ]

        instance = Model.model_validate(data)
        dumped = instance.model_dump()

        assert dumped[0] == "string"
        assert dumped[1] == 42
        assert dumped[3]["type"] == "custom"


class TestEdgeCasesAndValidation:
    """Test edge cases and validation scenarios."""

    def test_empty_schema(self) -> None:
        """Test handling of empty schema."""
        schema: dict[str, Any] = {}

        Model = create_model_from_json_schema(schema, "EmptySchemaModel")
        instance = Model.model_validate({})
        assert instance.model_dump() == {}

    def test_schema_with_no_properties(self) -> None:
        """Test object schema with no properties."""
        schema = {"type": "object", "properties": {}, "required": []}

        Model = create_model_from_json_schema(schema, "NoPropsModel")
        instance = Model.model_validate({})
        assert instance.model_dump() == {}

    def test_all_optional_fields_with_defaults(self) -> None:
        """Test schema where all fields are optional with defaults."""
        schema = {
            "type": "object",
            "properties": {
                "str_field": {"type": "string", "default": "default_string"},
                "int_field": {"type": "integer", "default": 42},
                "bool_field": {"type": "boolean", "default": True},
                "array_field": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
            },
            "required": [],
        }

        Model = create_model_from_json_schema(schema, "DefaultsModel")

        # Create with no data - should use defaults
        instance = Model.model_validate({})
        dumped = instance.model_dump(exclude_defaults=False)

        assert dumped["str_field"] == "default_string"
        assert dumped["int_field"] == 42
        assert dumped["bool_field"] is True
        assert dumped["array_field"] == []

    def test_required_fields_validation(self) -> None:
        """Test that required fields are actually enforced."""
        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"},
                "optional_field": {"type": "string"},
            },
            "required": ["required_field"],
        }

        Model = create_model_from_json_schema(schema, "RequiredValidationModel")

        # Should succeed with required field
        instance = Model.model_validate({"required_field": "value"})
        assert instance.required_field == "value"

        # Should fail without required field
        with pytest.raises(ValidationError):
            Model.model_validate({})

        with pytest.raises(ValidationError):
            Model.model_validate({"optional_field": "value"})

    def test_type_validation_enforcement(self) -> None:
        """Test that type validation is enforced."""
        schema = {
            "type": "object",
            "properties": {
                "str_field": {"type": "string"},
                "int_field": {"type": "integer"},
                "bool_field": {"type": "boolean"},
            },
            "required": ["str_field", "int_field"],
        }

        Model = create_model_from_json_schema(schema, "TypeValidationModel")

        # Valid data
        instance = Model.model_validate(
            {"str_field": "text", "int_field": 42, "bool_field": True}
        )
        assert instance.str_field == "text"

        # Invalid types should raise validation errors
        with pytest.raises(ValidationError):
            Model.model_validate(
                {"str_field": 123, "int_field": 42}  # Should be string
            )

    def test_nullable_fields_with_anyof(self) -> None:
        """Test fields that can be null using anyOf."""
        schema = {
            "type": "object",
            "properties": {
                "nullable_string": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "nullable_object": {
                    "anyOf": [
                        {"type": "object", "properties": {"value": {"type": "number"}}},
                        {"type": "null"},
                    ]
                },
            },
            "required": ["nullable_string"],
        }

        Model = create_model_from_json_schema(schema, "NullableFieldsModel")

        # Test with null values
        instance1 = Model.model_validate(
            {"nullable_string": None, "nullable_object": None}
        )
        assert instance1.nullable_string is None
        assert instance1.nullable_object is None

        # Test with actual values
        instance2 = Model.model_validate(
            {"nullable_string": "value", "nullable_object": {"value": 42.5}}
        )
        assert instance2.nullable_string == "value"
        assert instance2.nullable_object["value"] == 42.5


class TestRealWorldSchemas:
    """Test real-world complex schema patterns."""

    def test_openapi_style_response_schema(self) -> None:
        """Test OpenAPI-style response schema with nested references."""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "data": {
                    "type": "object",
                    "properties": {
                        "users": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "id": {"type": "string"},
                                    "username": {"type": "string"},
                                    "email": {"type": "string"},
                                    "profile": {
                                        "type": "object",
                                        "properties": {
                                            "bio": {"type": "string"},
                                            "avatar_url": {"type": "string"},
                                            "social_links": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                    },
                                    "is_active": {"type": "boolean"},
                                },
                                "required": ["id", "username", "email"],
                            },
                        },
                        "pagination": {
                            "type": "object",
                            "properties": {
                                "page": {"type": "integer"},
                                "per_page": {"type": "integer"},
                                "total": {"type": "integer"},
                            },
                            "required": ["page", "per_page", "total"],
                        },
                    },
                    "required": ["users"],
                },
                "meta": {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "string"},
                        "request_id": {"type": "string"},
                    },
                },
            },
            "required": ["status", "data"],
        }

        Model = create_model_from_json_schema(schema, "APIResponseModel")

        data = {
            "status": "success",
            "data": {
                "users": [
                    {
                        "id": "usr_123",
                        "username": "alice",
                        "email": "alice@example.com",
                        "profile": {
                            "bio": "Software Engineer",
                            "avatar_url": "https://example.com/avatar.jpg",
                            "social_links": ["https://twitter.com/alice"],
                        },
                        "is_active": True,
                    }
                ],
                "pagination": {"page": 1, "per_page": 10, "total": 100},
            },
            "meta": {"timestamp": "2025-01-01T00:00:00Z", "request_id": "req_abc123"},
        }

        instance = Model.model_validate(data)
        assert instance.status == "success"
        assert instance.data["users"][0]["username"] == "alice"
        assert instance.data["pagination"]["total"] == 100

    def test_configuration_schema_with_mixed_types(self) -> None:
        """Test configuration-style schema with various type combinations."""
        schema = {
            "type": "object",
            "properties": {
                "version": {"type": "string"},
                "database": {
                    "type": "object",
                    "properties": {
                        "host": {"type": "string"},
                        "port": {"type": "integer"},
                        "credentials": {
                            "anyOf": [
                                {
                                    "type": "object",
                                    "properties": {
                                        "username": {"type": "string"},
                                        "password": {"type": "string"},
                                    },
                                    "required": ["username", "password"],
                                },
                                {
                                    "type": "object",
                                    "properties": {"token": {"type": "string"}},
                                    "required": ["token"],
                                },
                            ]
                        },
                        "options": {
                            "type": "object",
                            "properties": {
                                "ssl": {"type": "boolean"},
                                "timeout": {"type": "integer"},
                                "retry_attempts": {"type": "integer"},
                            },
                        },
                    },
                    "required": ["host", "port"],
                },
                "features": {"type": "array", "items": {"type": "string"}},
                "limits": {
                    "type": "object",
                    "properties": {
                        "max_connections": {"type": "integer"},
                        "rate_limit": {
                            "anyOf": [{"type": "integer"}, {"type": "null"}]
                        },
                    },
                },
            },
            "required": ["version", "database"],
        }

        Model = create_model_from_json_schema(schema, "ConfigurationModel")

        # Test with username/password credentials
        config1 = {
            "version": "1.0.0",
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {"username": "admin", "password": "secret"},
                "options": {"ssl": True, "timeout": 30, "retry_attempts": 3},
            },
            "features": ["caching", "monitoring"],
            "limits": {"max_connections": 100, "rate_limit": 1000},
        }

        instance1 = Model.model_validate(config1)
        assert instance1.database["credentials"]["username"] == "admin"

        # Test with token credentials
        config2 = {
            "version": "2.0.0",
            "database": {
                "host": "db.example.com",
                "port": 3306,
                "credentials": {"token": "bearer_token_123"},
            },
            "features": [],
            "limits": {"max_connections": 50, "rate_limit": None},
        }

        instance2 = Model.model_validate(config2)
        assert instance2.database["credentials"]["token"] == "bearer_token_123"
        assert instance2.limits["rate_limit"] is None

    def test_empty_items_schema_in_arrays(self) -> None:
        """Test handling of array fields with empty items schema."""
        schema = {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The issue title",
                },
                "labels": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {},
                        },
                        {"type": "null"},
                    ],
                    "default": None,
                    "description": "Array of label names",
                },
                "links": {
                    "anyOf": [
                        {
                            "type": "array",
                            "items": {},
                        },
                        {"type": "null"},
                    ],
                    "default": None,
                    "description": "Array of link objects",
                },
                "team": {
                    "type": "string",
                    "description": "The team name",
                },
            },
            "required": ["title", "team"],
        }

        model = create_model_from_json_schema(schema, "EmptyItemsModel")
        assert issubclass(model, BaseModel)

        instance1 = model.model_validate(
            {
                "title": "Test Issue",
                "team": "engineering",
                "labels": None,
                "links": None,
            }
        )
        assert instance1.title == "Test Issue"
        assert instance1.labels is None

        instance2 = model.model_validate(
            {
                "title": "Test Issue 2",
                "team": "engineering",
                "labels": ["bug", "urgent", "backend"],
                "links": [
                    {"url": "https://example.com", "title": "Link 1"},
                    {"url": "https://example.com/2", "title": "Link 2"},
                ],
            }
        )
        assert instance2.labels == ["bug", "urgent", "backend"]
        assert len(instance2.links) == 2

        instance3 = model.model_validate(
            {
                "title": "Test Issue 3",
                "team": "engineering",
                "labels": ["string", 123, True, None, {"key": "value"}],
            }
        )
        assert len(instance3.labels) == 5

        dumped = instance2.model_dump()
        assert dumped["labels"] == ["bug", "urgent", "backend"]
        assert dumped["links"][0]["url"] == "https://example.com"
