from private_gpt.components.chat.models.chat_config_models import ToolSpec


def test_convert_into_function_tool_tool() -> None:
    tool = ToolSpec(
        name="test_tool",
        description="Test description",
        input_schema={
            "type": "object",
            "properties": {"param1": {"type": "string"}, "_param2": {"type": "number"}},
            "required": ["param1"],
        },
    )

    result = tool.to_function_tool()

    from llama_index.core.tools import FunctionTool

    assert isinstance(result, FunctionTool)

    assert result.metadata.name == "test_tool"
    assert result.metadata.description == "Test description"
    assert result.metadata.return_direct is True

    assert hasattr(result, "metadata")
    assert hasattr(result.metadata, "fn_schema")

    schema_model = result.metadata.fn_schema
    assert schema_model.__name__ == "test_tool_schema"

    schema_fields = schema_model.__fields__
    assert "param1" in schema_fields
    assert "param2" in schema_fields
    assert schema_fields["param1"].is_required()
    assert not schema_fields["param2"].is_required()


def test_model_serialization_without_alias() -> None:
    tool = ToolSpec(
        name="test_tool",
        description="Test description",
        input_schema={
            "type": "object",
            "properties": {
                "param1": {"type": "string"},
                "_param2": {"type": "number"},
                "class": {"type": "string"},
                "match": {"type": "string"},
            },
            "required": ["param1"],
        },
    )

    result = tool.to_function_tool()
    schema_model = result.metadata.fn_schema

    instance = schema_model(
        param1="test", _param2=42.0, **{"class": "test_class", "match": "test_match"}
    )

    # Explicitly request serialization without aliases
    serialized = instance.model_dump(by_alias=False)

    assert "param1" in serialized
    assert "param2" in serialized  # Sanitized name (leading underscore removed)
    assert "class_" in serialized  # Sanitized name (keyword + underscore)
    assert "match_" in serialized  # Sanitized name (soft keyword + underscore)

    # Original names should not be present
    assert "_param2" not in serialized
    assert "class" not in serialized
    assert "match" not in serialized


def test_model_serialization_with_original_schema() -> None:
    tool = ToolSpec(
        name="test_tool",
        description="Test description",
        input_schema={
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
        },
    )

    result = tool.to_function_tool()
    schema_model = result.metadata.fn_schema

    # Create instance with original field names (using aliases)
    instance = schema_model(
        param1="test_value",
        _param2=42.5,
        normal_param=True,
        **{"class": "test_class", "match": "test_match"},
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


def test_keyword_sanitization() -> None:
    """Test that Python keywords and soft keywords are properly sanitized."""
    tool = ToolSpec(
        name="test_tool",
        description="Test description",
        input_schema={
            "type": "object",
            "properties": {
                "def": {"type": "string"},
                "return": {"type": "string"},
                "case": {"type": "string"},
                "_": {"type": "string"},
            },
            "required": [],
        },
    )

    result = tool.to_function_tool()
    schema_model = result.metadata.fn_schema

    instance = schema_model(
        **{
            "def": "test_def",
            "return": "test_return",
            "case": "test_case",
            "_": "test_underscore",
        }
    )

    # Test serialization with aliases (should use original names)
    with_aliases = instance.model_dump(by_alias=True)
    assert "def" in with_aliases
    assert "return" in with_aliases
    assert "case" in with_aliases
    assert "_" in with_aliases

    # Test serialization without aliases (should use sanitized names)
    without_aliases = instance.model_dump(by_alias=False)
    assert "def_" in without_aliases
    assert "return_" in without_aliases
    assert "case_" in without_aliases
    assert "field" in without_aliases  # "_" becomes "field"


def test_field_name_collisions() -> None:
    """Test that field name collisions are resolved with counters."""
    tool = ToolSpec(
        name="test_tool",
        description="Test description",
        input_schema={
            "type": "object",
            "properties": {
                "": {"type": "string"},  # becomes "field"
                "   ": {"type": "string"},  # becomes "field_1"
                "123": {"type": "string"},  # becomes "field_2"
                "field": {"type": "string"},  # becomes "field_3"
                "_field": {"type": "string"},  # becomes "field_4"
            },
        },
    )

    result = tool.to_function_tool()
    schema_model = result.metadata.fn_schema

    instance = schema_model(
        **{
            "": "test1",
            "   ": "test2",
            "field": "test4",
            "_field": "test5",
        }
    )

    # Test serialization without aliases shows unique field names
    without_aliases = instance.model_dump(by_alias=False)
    assert "field" in without_aliases
    assert without_aliases["field"] == "test1"
    assert "field_1" in without_aliases
    assert without_aliases["field_1"] == "test2"
    assert "field_2" in without_aliases
    assert without_aliases["field_2"] == "test4"
    assert "field_3" in without_aliases
    assert without_aliases["field_3"] == "test5"
    assert len(without_aliases) == 5  # All fields are unique
