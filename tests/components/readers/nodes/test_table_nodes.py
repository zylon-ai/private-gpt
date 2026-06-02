import json
import re
import uuid
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from private_gpt.components.ingest.metadata_helper import MetadataKeys
from private_gpt.components.ingest.processors.df_preprocessor import (
    DataFramePreprocessor,
)
from private_gpt.components.readers.nodes import TextNode
from private_gpt.components.readers.nodes.partial_node import PartialNode
from private_gpt.components.readers.nodes.table_node import TableNode, TableRowNode
from private_gpt.components.readers.nodes.tree_node import (
    MetadataMode,
    TreeMetadataMode,
)

if TYPE_CHECKING:
    from private_gpt.components.readers.nodes.tree_node import TreeNode

mock_extra_info = {
    MetadataKeys.ARTIFACT_ID.value: str(uuid.uuid4()),
    MetadataKeys.COLLECTION.value: str(uuid.uuid4()),
}


@pytest.fixture
def valid_dataframe() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": [25, 30],
            "Birthday": ["1996-01-01", "1991-01-01"],
        }
    )
    df = DataFramePreprocessor().preprocess_table(df)
    return df


@pytest.fixture
def valid_tablerow_data(valid_dataframe: pd.DataFrame) -> dict[str, Any]:
    return {
        "header": list(valid_dataframe.columns),
        "content": valid_dataframe.iloc[0].tolist(),
    }


def test_valid_table_node(valid_dataframe: pd.DataFrame) -> None:
    node = TableNode(df=valid_dataframe, description="Test Table")
    assert node.df.equals(valid_dataframe)
    assert node.description == "Test Table"


def test_empty_dataframe() -> None:
    with pytest.raises(ValueError, match=re.escape("Empty dataframe provided.")):
        TableNode(df=pd.DataFrame()).set_content(
            TableNode.Meta(dataframe=pd.DataFrame(), summary=None)
        )


def test_table_node_serialization(valid_dataframe: pd.DataFrame) -> None:
    node = TableNode(df=valid_dataframe, description="Serialization Test")
    serialized = node.dict()
    json_str = json.dumps(serialized)
    json_obj = json.loads(json_str)
    deserialized = TableNode.from_dict(json_obj)
    assert deserialized.df.equals(valid_dataframe)
    assert deserialized.description == "Serialization Test"


def test_table_node_serialization_with_non_serializable_data(
    valid_dataframe: pd.DataFrame,
) -> None:
    # replace valid dataframe with NA and NaT values
    valid_dataframe = pd.DataFrame(
        {
            "Name": ["Alice", "Bob", pd.NA],
            "Age": [25, 30, np.NaN],
            "Birthday": ["1996-01-01", "1991-01-01", pd.NaT],
        }
    )
    node = TableNode(df=valid_dataframe, description="Serialization Test")
    serialized = node.dict()
    json_str = json.dumps(serialized)
    json_obj = json.loads(json_str)
    deserialized = TableNode.from_dict(json_obj)
    assert deserialized


def test_get_content_all_metadata(valid_dataframe: pd.DataFrame) -> None:
    node = TableNode(
        df=valid_dataframe, description="Metadata Test", extra_info=mock_extra_info
    )
    content = node.get_content(metadata_mode=MetadataMode.ALL)
    assert mock_extra_info[MetadataKeys.ARTIFACT_ID.value] in content
    assert mock_extra_info[MetadataKeys.COLLECTION.value] in content
    assert "Name" in content
    assert "25" in content


def test_get_content_no_metadata(valid_dataframe: pd.DataFrame) -> None:
    node = TableNode(
        df=valid_dataframe, description="Metadata Test", extra_info=mock_extra_info
    )
    content = node.get_content(metadata_mode=MetadataMode.NONE)
    assert mock_extra_info[MetadataKeys.ARTIFACT_ID.value] not in content
    assert mock_extra_info[MetadataKeys.COLLECTION.value] not in content
    assert "|" in content.strip()


def test_get_content_llm(valid_dataframe: pd.DataFrame) -> None:
    node = TableNode(df=valid_dataframe, extra_info=mock_extra_info)
    row = TableRowNode(header=["Name", "Age"], content=["Alice", 25])
    node.add_child(row)
    content = node.get_content(metadata_mode=MetadataMode.LLM)
    assert mock_extra_info[MetadataKeys.ARTIFACT_ID.value] in content
    assert mock_extra_info[MetadataKeys.COLLECTION.value] in content
    assert "Alice" in content.strip()
    assert "Table description" not in content


def test_get_content_embed(valid_dataframe: pd.DataFrame) -> None:
    node = TableNode(
        df=valid_dataframe, description="Metadata Test", extra_info=mock_extra_info
    )
    content = node.get_content(metadata_mode=MetadataMode.EMBED)
    assert mock_extra_info[MetadataKeys.ARTIFACT_ID.value] in content
    assert mock_extra_info[MetadataKeys.COLLECTION.value] in content
    assert "|" in content.strip()
    assert "Table description" in content


def test_mixed_data_types() -> None:
    df = pd.DataFrame(
        {
            "Numbers": [1, 2, pd.NA],
            "Strings": ["A", pd.NA, "C"],
            "Booleans": [True, False, pd.NA],
        }
    )
    df = df.convert_dtypes()
    node = TableNode(df=df, description="Mixed Data")
    content = node.get_content(metadata_mode=MetadataMode.ALL)
    assert "<NA>" not in content
    assert "Numbers" in content
    assert "A" in content


def test_special_characters() -> None:
    df = pd.DataFrame(
        {
            "Name": ["<Alice>", "Bob & Carol"],
            "Bio": ["Loves *coding*", "Enjoys `Python`"],
        }
    )
    node = TableNode(df=df, description="Special Characters")
    content = node.get_content(metadata_mode=MetadataMode.ALL)
    assert "<Alice>" in content
    assert "*coding*" in content
    assert "`Python`" in content


def test_valid_tablerow_node(valid_tablerow_data: dict[str, Any]) -> None:
    node = TableRowNode(
        header=valid_tablerow_data["header"], content=valid_tablerow_data["content"]
    )
    assert node.header == valid_tablerow_data["header"]
    assert node.content == valid_tablerow_data["content"]


def test_tablerow_header_content_length_mismatch() -> None:
    with pytest.raises(ValueError, match="Header and content length mismatch"):
        TableRowNode(header=[], content=[]).set_content(
            TableRowNode.Meta(header=["Name", "Age"], content=["Alice"])
        )


def test_tablerow_serialization(valid_tablerow_data: dict[str, Any]) -> None:
    node = TableRowNode(
        header=valid_tablerow_data["header"], content=valid_tablerow_data["content"]
    )
    serialized = node.dict()
    json_str = json.dumps(serialized)
    json_obj = json.loads(json_str)
    deserialized = TableRowNode.from_dict(json_obj)
    assert deserialized.header == valid_tablerow_data["header"]
    assert deserialized.content == valid_tablerow_data["content"]


def test_tablerow_get_content_all_metadata(valid_tablerow_data: dict[str, Any]) -> None:
    node = TableRowNode(
        header=valid_tablerow_data["header"], content=valid_tablerow_data["content"]
    )
    content = node.get_content(metadata_mode=MetadataMode.ALL)
    assert "Alice" in content
    assert "25" in content
    # Check that it's markdown
    assert "|" in content.strip()


def test_tablerow_get_content_no_metadata(valid_tablerow_data: dict[str, Any]) -> None:
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": [25, 30],
        }
    )
    node = TableNode(df=df)
    row_1 = TableRowNode(header=["Name", "Age"], content=["Alice", 25])
    row_2 = TableRowNode(header=["Name", "Age"], content=["Bob", 30])
    node.add_child(row_1)
    node.add_child(row_2)

    # Check that parent returns full table
    content = node.get_content(metadata_mode=MetadataMode.NONE)
    assert "Name" in content
    assert "Alice" in content
    assert "|" in content.strip()
    assert content.count("Alice") == 1

    # Check that children return markdown
    content = row_1.get_content(metadata_mode=MetadataMode.NONE)
    assert "Name" not in content
    assert "Alice" in content
    assert "25" in content
    assert "|" in content.strip()
    assert content.count("Alice") == 1

    content = row_2.get_content(metadata_mode=TreeMetadataMode.NONE)
    assert "Name" not in content
    assert "Bob" in content
    assert "30" in content
    assert "|" in content.strip()
    assert content.count("Bob") == 1


def test_get_content_user_metadata(valid_tablerow_data: dict[str, Any]) -> None:
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": [25, 30],
        }
    )
    node = TableNode(df=df)
    row_1 = TableRowNode(header=list(df.columns), content=["Alice", 25])
    row_2 = TableRowNode(header=list(df.columns), content=["Bob", 30])
    node.add_child(row_1)
    node.add_child(row_2)
    node.update_references()

    # Check that parent returns only one table
    content = node.get_content(metadata_mode=TreeMetadataMode.USER)
    assert "Name" in content
    assert "Alice" in content
    assert "25" in content
    assert "|" in content.strip()
    assert content.count("Alice") == 1

    # Check that children return markdown
    content = row_1.get_content(metadata_mode=TreeMetadataMode.USER)
    assert "Name" in content
    assert "Alice" in content
    assert "25" in content
    assert "|" in content.strip()
    assert content.count("Alice") == 1

    content = row_2.get_content(metadata_mode=TreeMetadataMode.USER)
    assert "Name" not in content
    assert "Bob" in content
    assert "30" in content
    assert "|" in content.strip()
    assert content.count("Bob") == 1


def test_get_content_llm_metadata(valid_tablerow_data: dict[str, Any]) -> None:
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": [25, 30],
        }
    )
    node = TableNode(df=df)
    row_1 = TableRowNode(header=list(df.columns), content=["Alice", 25])
    row_2 = TableRowNode(header=list(df.columns), content=["Bob", 30])
    node.add_child(row_1)
    node.add_child(row_2)
    node.update_references()

    # Check that parent returns only one table
    content = node.get_content(metadata_mode=TreeMetadataMode.LLM)
    assert "Alice" in content
    assert "25" in content
    assert "|" in content.strip()
    assert content.count("Alice") == 1

    # Check that children return markdown
    content = row_1.get_content(metadata_mode=TreeMetadataMode.LLM)
    assert "Name" in content
    assert "Alice" in content
    assert "25" in content
    assert "|" in content.strip()
    assert content.count("Alice") == 1

    content = row_2.get_content(metadata_mode=TreeMetadataMode.LLM)
    assert "Name" not in content
    assert "Bob" in content
    assert "30" in content
    assert "|" in content.strip()
    assert content.count("Bob") == 1


def test_get_content_embed_metadata(valid_tablerow_data: dict[str, Any]) -> None:
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": [25, 30],
        }
    )
    node = TableNode(df=df)
    row_1 = TableRowNode(header=list(df.columns), content=["Alice", 25])
    row_2 = TableRowNode(header=list(df.columns), content=["Bob", 30])
    node.add_child(row_1)
    node.add_child(row_2)
    node.update_references()

    # Check that parent returns only one table
    content = node.get_content(metadata_mode=TreeMetadataMode.EMBED)
    assert "Alice" in content
    assert "25" in content
    assert "|" in content.strip()
    assert content.count("Alice") == 1

    # Check that children return markdown
    content = row_1.get_content(metadata_mode=TreeMetadataMode.EMBED)
    assert "Name" in content
    assert "Alice" in content
    assert "25" in content
    assert "|" not in content.strip()
    assert ":" in content.strip()
    assert content.count("Alice") == 1

    content = row_2.get_content(metadata_mode=TreeMetadataMode.EMBED)
    assert "Name" in content
    assert "Bob" in content
    assert "30" in content
    assert "|" not in content.strip()
    assert ":" in content.strip()


def test_is_first_row() -> None:
    # Case 1.: Using index
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": [25, 30],
        }
    )
    node = TableNode(df=df)
    row_1: TreeNode = TableRowNode(header=list(df.columns), content=["Alice", 25])
    row_2: TreeNode = TableRowNode(header=list(df.columns), content=["Bob", 30])
    node.add_child(row_1)
    node.add_child(row_2)
    node.update_references()

    assert row_1.is_first_row()
    assert not row_2.is_first_row()

    # Case 2. Using left sibling
    node = TableNode(df=df)
    row_1 = PartialNode.from_node(row_1)
    node.add_child(row_1)
    node.add_child(row_2)

    assert row_2.is_first_row()


def test_is_last_row() -> None:
    # Case 1.: Using index
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": [25, 30],
        }
    )
    node = TableNode(df=df)
    row_1: TreeNode = TableRowNode(header=list(df.columns), content=["Alice", 25])
    row_2: TreeNode = TableRowNode(header=list(df.columns), content=["Bob", 30])
    node.add_child(row_1)
    node.add_child(row_2)
    node.update_references()

    assert row_2.is_last_row()
    assert not row_1.is_last_row()

    # Case 2. Using right sibling
    node = TableNode(df=df)
    row_2 = PartialNode.from_node(row_2)
    node.add_child(row_1)
    node.add_child(row_2)

    assert row_1.is_last_row()


def test_representation() -> None:
    # Case 1. Only one row
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": [25, 30],
        }
    )
    node = TableNode(df=df)
    row_1: TreeNode = TableRowNode(header=list(df.columns), content=["Alice", 25])
    node.add_child(row_1, update_references=True)

    content = row_1.get_content(metadata_mode=TreeMetadataMode.RAG)
    assert " - " in content
    assert content.count("\n") == 4

    # Case 2. Multiple rows
    row_2: TreeNode = TableRowNode(header=list(df.columns), content=["Bob", 30])
    node.add_child(row_2, update_references=True)

    content = row_1.get_content(metadata_mode=TreeMetadataMode.RAG)
    assert " - " in content
    assert content.count("\n") == 4
    content = row_2.get_content(metadata_mode=TreeMetadataMode.RAG)
    assert " - " not in content
    assert content.count("\n") == 2

    # Case 3. Add a before and after partial node
    before_partial = PartialNode(type=TableRowNode.get_type())
    after_partial = PartialNode(type=TableRowNode.get_type())
    node.insert_child(0, before_partial)
    node.add_child(after_partial, update_references=True)

    content = row_1.get_content(metadata_mode=TreeMetadataMode.RAG)
    assert " - " in content
    assert content.count("\n") == 4
    content = row_2.get_content(metadata_mode=TreeMetadataMode.RAG)
    assert " - " not in content
    assert content.count("\n") == 2


def test_tablerow_empty_header_and_content() -> None:
    # Case 1. Empty header and content
    node = TableRowNode(header=[], content=[])
    content = node.get_content(metadata_mode=MetadataMode.ALL)
    assert content.strip() == ""

    # Case 2. Empty header
    node = TableRowNode(header=[], content=["Alice", 25])
    content = node.get_content(metadata_mode=MetadataMode.ALL)
    assert content.strip() == ""

    # Case 3. Empty content
    node = TableRowNode(header=["Name", "Age"], content=[])
    content = node.get_content(metadata_mode=MetadataMode.ALL)
    assert content.strip() == ""


def test_tablerow_mixed_data_types() -> None:
    node = TableRowNode(
        header=["Numbers", "Strings", "Booleans"], content=[pd.NA, "A", True]
    )
    content = node.get_content(metadata_mode=MetadataMode.ALL)
    assert "<NA>" not in content
    assert "A" in content
    assert "True" in content


def test_tablerow_special_characters() -> None:
    node = TableRowNode(header=["Name"], content=["<Alice>"])
    content = node.get_content(metadata_mode=MetadataMode.ALL)
    assert "<Alice>" in content


def test_minimal_markdown_table() -> None:
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": [25, 30],
        }
    )
    node = TableNode(df=df)
    content = node.get_content(metadata_mode=MetadataMode.ALL)
    expected_output = (
        "| Name | Age |\n" "| - | - |\n" "| Alice | 25 |\n" "| Bob | 30 |\n\n"
    )
    assert content.strip() == expected_output.strip()


def test_table_row_position_with_partials() -> None:
    df = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [25, 30]})
    table = TableNode(df=df)

    # Pattern: partial, row, row, partial
    partial1 = PartialNode(type=TableRowNode.get_type())
    row1 = TableRowNode(header=list(df.columns), content=["Alice", 25])
    row2 = TableRowNode(header=list(df.columns), content=["Bob", 30])
    partial2 = PartialNode(type=TableRowNode.get_type())

    table.add_child(partial1)
    table.add_child(row1)
    table.add_child(row2)
    table.add_child(partial2)
    table.update_references()

    assert row1.is_first_row()
    assert not row1.is_last_row()
    assert not row2.is_first_row()
    assert row2.is_last_row()


def test_table_row_single_among_partials() -> None:
    df = pd.DataFrame({"Name": ["Alice"], "Age": [25]})
    table = TableNode(df=df)

    # Pattern: partial, partial, row, partial, partial
    for _ in range(2):
        table.add_child(PartialNode(type=TableRowNode.get_type()))

    row = TableRowNode(header=list(df.columns), content=["Alice", 25])
    table.add_child(row)

    for _ in range(2):
        table.add_child(PartialNode(type=TableRowNode.get_type()))

    table.update_references()

    assert row.is_first_row()
    assert row.is_last_row()


def test_table_row_corrupted_index() -> None:
    df = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [25, 30]})
    table = TableNode(df=df)

    row1 = TableRowNode(header=list(df.columns), content=["Alice", 25])
    row2 = TableRowNode(header=list(df.columns), content=["Bob", 30])
    table.add_child(row1)
    table.add_child(row2)
    table.update_references()

    # Corrupt the indices
    row1.idx = 1
    row2.idx = 0

    # Should still work via search fallback
    assert row1.is_first_row()
    assert not row1.is_last_row()
    assert not row2.is_first_row()
    assert row2.is_last_row()


def test_table_row_no_parent_or_siblings() -> None:
    row = TableRowNode(header=["Name"], content=["Alice"])

    # No parent
    assert not row.is_first_row()
    assert not row.is_last_row()

    # Parent with empty children
    table = TableNode(df=pd.DataFrame({"Name": ["Alice"]}))
    row.parent = table
    table.children = []

    assert not row.is_first_row()
    assert not row.is_last_row()


def test_table_row_serialization_with_numpy_types() -> None:
    content = [
        np.int64(42),
        np.float32(3.14),
        np.str_("test"),
        np.bool_(True),
        np.datetime64("2024-01-01"),
    ]
    headers = ["int", "float", "str", "bool", "date"]

    row = TableRowNode(header=headers, content=content)
    serialized = row.model_dump()
    deserialized = TableRowNode.from_dict(serialized)

    assert deserialized.header == headers
    assert len(deserialized.content) == len(content)


def test_table_row_missing_values_handling() -> None:
    content = [1, pd.NA, np.nan, None, pd.NaT]
    headers = ["int", "pd_na", "np_nan", "none", "nat"]

    row = TableRowNode(header=headers, content=content)

    # Test content generation doesn't crash
    content_str = row.get_content(metadata_mode=TreeMetadataMode.ALL)
    assert "1" in content_str

    # Test serialization round trip
    serialized = row.model_dump()
    deserialized = TableRowNode.from_dict(serialized)
    assert deserialized.header == headers


def test_table_row_embed_mode_key_value_format() -> None:
    df = pd.DataFrame({"Name": ["Alice", "Bob"], "Age": [25, 30]})
    table = TableNode(df=df)

    row = TableRowNode(header=list(df.columns), content=["Alice", 25])
    table.add_child(row)
    table.update_references()

    content = row.get_content(metadata_mode=TreeMetadataMode.EMBED)

    # Should be key-value format, not markdown
    assert "Name: Alice" in content
    assert "Age: 25" in content
    assert "|" not in content  # No markdown table format


def test_table_row_header_footer_formatting() -> None:
    df = pd.DataFrame({"Name": ["Alice", "Bob", "Charlie"], "Age": [25, 30, 35]})
    table = TableNode(df=df)

    rows = []
    for i in range(3):
        row = TableRowNode(header=list(df.columns), content=df.iloc[i].tolist())
        table.add_child(row)
        rows.append(row)

    table.update_references()

    # First row should include header
    first_content = rows[0].get_content(metadata_mode=TreeMetadataMode.RAG)
    assert first_content.count("Name") >= 1
    assert first_content.count("\n") == 4  # Header + separator + data + footer

    # Middle row should not include header
    middle_content = rows[1].get_content(metadata_mode=TreeMetadataMode.RAG)
    assert "Name" not in middle_content
    assert middle_content.count("\n") == 1  # Just data

    # Last row should include footer
    last_content = rows[2].get_content(metadata_mode=TreeMetadataMode.RAG)
    assert last_content.count("\n") == 2  # Data + footer


def test_table_node_add_row_functionality() -> None:
    df = pd.DataFrame({"Name": ["Alice"], "Age": [25]})
    table = TableNode(df=df.copy())

    original_length = len(table.df)

    # Add compatible row
    table.add_row(["Bob", 30])
    assert len(table.df) == original_length + 1
    assert table.df.iloc[-1]["Name"] == "Bob"
    assert table.df.iloc[-1]["Age"] == 30

    # Test incompatible row length
    with pytest.raises(ValueError, match="Row length mismatch"):
        table.add_row(["Too", "Many", "Values", "Here"])


def test_table_node_row_compatibility() -> None:
    df = pd.DataFrame({"Name": ["Alice"], "Age": [25], "City": ["NYC"]})
    table = TableNode(df=df)

    # Compatible row
    compatible_row = TableRowNode(
        header=["Name", "Age", "City"], content=["Bob", 30, "SF"]
    )
    assert table.is_row_compatible(compatible_row)

    # Incompatible - different columns
    incompatible_row = TableRowNode(
        header=["FirstName", "LastName"], content=["John", "Doe"]
    )
    assert not table.is_row_compatible(incompatible_row)

    # Incompatible - different order
    wrong_order_row = TableRowNode(
        header=["Age", "Name", "City"], content=[25, "Alice", "NYC"]
    )
    assert not table.is_row_compatible(wrong_order_row)


def test_table_row_special_characters() -> None:
    special_content = [
        "text|with|pipes",
        "text\nwith\nnewlines",
        "<html>tags</html>",
        "text&with&ampersands",
        'text"with"quotes',
        "  text with spaces  ",
    ]
    headers = [f"col_{i}" for i in range(len(special_content))]

    row = TableRowNode(header=headers, content=special_content)
    content = row.get_content(metadata_mode=TreeMetadataMode.LLM)

    # Should preserve special characters
    assert "text|with|pipes" in content
    assert "<html>tags</html>" in content
    assert "text&with&ampersands" in content


def test_table_row_empty_content_edge_cases() -> None:
    # Empty header and content
    empty_row = TableRowNode(header=[], content=[])
    content = empty_row.get_content(metadata_mode=TreeMetadataMode.ALL)
    assert content.strip() == ""

    # Mismatched empty content
    mismatch_row = TableRowNode(header=["Name"], content=[])
    content = mismatch_row.get_content(metadata_mode=TreeMetadataMode.ALL)
    assert content.strip() == ""

    # Empty header with content
    no_header_row = TableRowNode(header=[], content=["Alice", 25])
    content = no_header_row.get_content(metadata_mode=TreeMetadataMode.ALL)
    assert content.strip() == ""


@pytest.mark.parametrize("table_size", [1_000, 10_000, 100_000, 1_000_000])
def test_million_row_simulation(table_size: int) -> None:
    df = pd.DataFrame({"id": [1], "name": ["test"]})
    table = TableNode(df=df)

    # Many PartialNodes simulating data
    size = table_size // 2
    for _ in range(size - 1):
        partial = PartialNode(type=TableRowNode.get_type())
        table.add_child(partial)

    # First actual row
    first_row = TableRowNode(header=["id", "name"], content=[1, "first"])
    table.add_child(first_row)

    # Last actual row
    last_row = TableRowNode(header=["id", "name"], content=[table_size, "last"])
    table.add_child(last_row)

    # Many PartialNodes simulating data
    for _ in range(size - 1):
        partial = PartialNode(type=TableRowNode.get_type())
        table.add_child(partial)

    table.update_references()

    assert len(table.children) == table_size
    assert first_row.is_first_row()
    assert not first_row.is_last_row()
    assert not last_row.is_first_row()
    assert last_row.is_last_row()


def test_table_row_mixed_node_types_in_table() -> None:
    df = pd.DataFrame({"Name": ["Alice"], "Age": [25]})
    table = TableNode(df=df)

    # Mix of TableRowNodes and other PartialNodes
    text_partial = PartialNode(type=TextNode.get_type())
    row1 = TableRowNode(header=list(df.columns), content=["Alice", 25])
    another_partial = PartialNode(type=TextNode.get_type())
    row2 = TableRowNode(header=list(df.columns), content=["Bob", 30])

    table.add_child(text_partial)
    table.add_child(row1)
    table.add_child(row2)
    table.add_child(another_partial)
    table.update_references()

    # Only TableRowNodes should be considered for first/last detection
    assert row1.is_first_row()
    assert not row1.is_last_row()
    assert not row2.is_first_row()
    assert row2.is_last_row()
