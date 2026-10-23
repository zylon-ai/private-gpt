from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
from llama_index.core.schema import MetadataMode
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
)

from private_gpt.components.readers.nodes.document_node import DocumentRootNode
from private_gpt.components.readers.nodes.list_node import ListItemNode, ListNode
from private_gpt.components.readers.nodes.partial_node import PartialNode
from private_gpt.components.readers.nodes.section_node import SectionNode
from private_gpt.components.readers.nodes.table_node import TableNode, TableRowNode
from private_gpt.components.readers.nodes.text_node import TextNode
from private_gpt.components.readers.nodes.tree_node import (
    CURRENT_VERSION,
    TreeMetadataMode,
)
from private_gpt.components.readers.nodes.utils import combine_trees
from private_gpt.components.readers.nodes.v2.document_node_v2 import DocumentRootNodeV2

if TYPE_CHECKING:
    from private_gpt.components.readers.nodes.tree_node import TreeNode


def test_document_tree_structure() -> None:
    # Create a document tree with multiple node types
    root = DocumentRootNode()

    # Add a section
    section1 = SectionNode(text="Introduction")
    root.add_child(section1)

    # Add text to section
    text_node = TextNode(text="This is an introduction paragraph.")
    section1.add_child(text_node)

    # Create a table
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": [30, 25],
            "City": ["New York", "San Francisco"],
        }
    )
    table = TableNode(df=df, description="Sample table")
    root.add_child(table)

    # Add table rows
    row1 = TableRowNode(header=df.columns.tolist(), content=df.iloc[0].tolist())
    table.add_child(row1)
    row2 = TableRowNode(header=df.columns.tolist(), content=df.iloc[1].tolist())
    table.add_child(row2)

    # Test tree structure
    assert len(root.children) == 2
    assert root.children[0] == section1
    assert root.children[1] == table

    assert len(section1.children) == 1
    assert section1.children[0] == text_node

    assert len(table.children) == 2
    assert table.children[0] == row1
    assert table.children[1] == row2


def test_node_serialization() -> None:
    # Create a sample document tree
    root = DocumentRootNode()
    section = SectionNode(text="Test Section")
    text = TextNode(text="Sample text")

    root.add_child(section)
    section.add_child(text)

    # Test serialization
    serialized = root.dict()
    assert serialized["class_name"] == f"DocumentRootNode-{CURRENT_VERSION}"
    assert "parent" not in serialized
    assert "children" not in serialized

    serialized = section.dict()
    assert serialized["class_name"] == f"SectionNode-{CURRENT_VERSION}"
    assert serialized["text"] == "Test Section"
    assert serialized["root_id"] == root.id_

    serialized = text.dict()
    assert serialized["class_name"] == f"TextNode-{CURRENT_VERSION}"
    assert serialized["text"] == "Sample text"
    assert serialized["root_id"] == root.id_


def test_node_serialization_utils() -> None:
    # Create a sample document tree
    root = DocumentRootNode()

    # Create a section
    section = SectionNode(text="Test Section")
    text = TextNode(text="Sample text")

    root.add_child(section)
    section.add_child(text)

    # Create a table
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": [30, 25],
            "City": ["New York", "San Francisco"],
        }
    )
    table = TableNode(df=df, description="Sample table")
    section.add_child(table)

    # Add table rows
    row1 = TableRowNode(header=df.columns.tolist(), content=df.iloc[0].tolist())
    table.add_child(row1)
    row2 = TableRowNode(header=df.columns.tolist(), content=df.iloc[1].tolist())
    table.add_child(row2)

    # Test document root serialization
    serialized = node_to_metadata_dict(root)
    assert serialized["_node_type"] == f"DocumentRootNode-{CURRENT_VERSION}"

    # Test section serialization
    serialized = node_to_metadata_dict(section)
    assert serialized["_node_type"] == f"SectionNode-{CURRENT_VERSION}"

    # Test table serialization
    serialized = node_to_metadata_dict(table)
    assert serialized["_node_type"] == f"TableNode-{CURRENT_VERSION}"


def test_node_deserialization() -> None:
    # Create a sample document tree
    root = DocumentRootNode()
    section = SectionNode(text="Test Section")
    text = TextNode(text="Sample text")

    root.add_child(section)
    section.add_child(text)

    # Serialize and deserialize
    serialized = root.dict()
    doc_deserialized = DocumentRootNode.parse_obj(serialized)
    assert doc_deserialized.id_ == root.id_
    assert doc_deserialized.hash == root.hash
    assert doc_deserialized.idx == root.idx
    assert doc_deserialized.abs_idx == root.abs_idx
    assert doc_deserialized.depth == root.depth
    assert doc_deserialized.height == root.height

    serialized = section.dict()
    section_deserialized = SectionNode.parse_obj(serialized)
    assert section_deserialized.hash == section.hash
    assert section_deserialized.text == section.text
    assert section_deserialized.parent_id == section.parent_id
    assert section_deserialized.root_id == section.root_id
    assert section_deserialized.idx == section.idx
    assert section_deserialized.abs_idx == section.abs_idx
    assert section_deserialized.depth == section.depth
    assert section_deserialized.height == section.height

    serialized = text.dict()
    text_deserialized = TextNode.parse_obj(serialized)
    assert text_deserialized.hash == text.hash
    assert text_deserialized.text == text.text
    assert text_deserialized.parent_id == text.parent_id
    assert text_deserialized.root_id == text.root_id
    assert text_deserialized.idx == text.idx
    assert text_deserialized.abs_idx == text.abs_idx
    assert text_deserialized.depth == text.depth
    assert text_deserialized.height == text.height


def test_node_deserialization_utils() -> None:
    # Create a sample document tree
    root = DocumentRootNode()

    # Create a section
    section = SectionNode(text="Test Section")
    text = TextNode(text="Sample text")

    root.add_child(section)
    section.add_child(text)

    # Create a table
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": [30, 25],
            "City": ["New York", "San Francisco"],
        }
    )
    table = TableNode(df=df, description="Sample table")
    section.add_child(table)

    # Add table rows
    row1 = TableRowNode(header=df.columns.tolist(), content=df.iloc[0].tolist())
    table.add_child(row1)
    row2 = TableRowNode(header=df.columns.tolist(), content=df.iloc[1].tolist())
    table.add_child(row2)

    # Test document root serialization
    serialized = node_to_metadata_dict(root)
    json = root.model_dump_json()
    node_content: str = serialized.get("_node_content") or ""
    assert node_content
    node_type: str = serialized.get("_node_type") or ""
    assert node_type == f"DocumentRootNode-{CURRENT_VERSION}"
    doc_obj = DocumentRootNode.from_json(node_content)
    doc_json = doc_obj.model_dump_json()
    assert doc_obj.hash == root.hash
    assert doc_obj.idx == root.idx
    assert doc_obj.abs_idx == root.abs_idx
    assert doc_obj.depth == root.depth
    assert doc_obj.height == root.height
    assert doc_json == json

    # Test section serialization
    serialized = node_to_metadata_dict(section)
    json = root.model_dump_json()
    node_content = serialized.get("_node_content") or ""
    assert node_content
    node_type = serialized.get("_node_type") or ""
    assert node_type == f"SectionNode-{CURRENT_VERSION}"
    section_obj = SectionNode.from_json(node_content)
    section_json = doc_obj.model_dump_json()
    assert section_obj.hash == section.hash
    assert section_obj.text == section.text
    assert section_obj.parent_id == section.parent_id
    assert section_obj.root_id == section.root_id
    assert section_obj.idx == section.idx
    assert section_obj.abs_idx == section.abs_idx
    assert section_obj.depth == section.depth
    assert section_obj.height == section.height
    assert json == section_json

    # Test table serialization
    serialized = node_to_metadata_dict(table)
    json = table.model_dump_json()
    node_content = serialized.get("_node_content") or ""
    assert node_content
    node_type = serialized.get("_node_type") or ""
    assert node_type == f"TableNode-{CURRENT_VERSION}"
    table_obj = TableNode.from_json(node_content)
    table_json = table_obj.model_dump_json()
    assert table_obj.hash == table.hash
    assert table_obj.description == table.description
    assert table_obj.df.equals(table.df)
    assert table_obj.parent_id == table.parent_id
    assert table_obj.root_id == table.root_id
    assert table_obj.idx == table.idx
    assert table_obj.abs_idx == table.abs_idx
    assert table_obj.depth == table.depth
    assert table_obj.height == table.height
    assert json == table_json


def test_node_flattening() -> None:
    # Create a nested document tree
    root = DocumentRootNode()
    section1 = SectionNode(text="Section 1")
    section2 = SectionNode(text="Section 2")
    text1 = TextNode(text="Text in Section 1")
    text2 = TextNode(text="Text in Section 2")

    root.add_child(section1)
    root.add_child(section2)
    section1.add_child(text1)
    section2.add_child(text2)

    # Test flattening
    flattened_nodes = list(root.flatten())
    assert len(flattened_nodes) == 5  # root + 2 sections + 2 texts
    assert flattened_nodes[0] == root
    assert flattened_nodes[1] == section1
    assert flattened_nodes[2] == text1
    assert flattened_nodes[3] == section2
    assert flattened_nodes[4] == text2


def test_node_metadata() -> None:
    # Test metadata functionality
    text_node = TextNode(text="Hello World")
    text_node.metadata["source"] = "test"

    # Test metadata string generation
    default_metadata_str = text_node.get_metadata_str()
    assert "source: test" in default_metadata_str

    # Test LLM metadata exclusion
    text_node.excluded_llm_metadata_keys = ["source"]
    llm_metadata_str = text_node.get_metadata_str(mode=MetadataMode.LLM)
    assert "source: test" not in llm_metadata_str


def test_node_hash() -> None:
    # Ensure hash is consistent for same node
    node1 = TextNode(text="Test")
    node2 = TextNode(text="Test")

    assert node1.hash == node2.hash

    # Ensure different nodes have different hashes
    node3 = TextNode(text="Different")
    assert node1.hash != node3.hash


def test_parent_child_relationship() -> None:
    root = DocumentRootNode()
    section = SectionNode(text="Test")
    text = TextNode(text="Content")

    root.add_child(section)
    section.add_child(text)

    # Test parent-child relationships
    assert text.parent == section
    assert section.parent == root
    assert root.parent is None

    assert text in section.children
    assert section in root.children


def test_document_node_tree_serialization() -> None:
    root = DocumentRootNode()
    section = SectionNode(text="Test")
    text = TextNode(text="Content")

    root.add_child(section)
    section.add_child(text)

    json = root.to_tree_serialization()
    assert json

    deserialized = DocumentRootNode()
    deserialized.from_tree_serialization(json)
    assert len(deserialized.children) == 1
    assert len(deserialized.children[0].children) == 1


@pytest.mark.parametrize("num_nodes", [1000, 10000, 100000])
def test_document_node_v1_tree_serialization(num_nodes: int) -> None:
    root = DocumentRootNode()
    num_nodes_by_type = num_nodes // 3
    text_num_nodes, table_num_nodes, list_num_nodes = (
        num_nodes_by_type,
        num_nodes_by_type,
        num_nodes_by_type,
    )

    texts: list[TreeNode] = []
    for i in range(text_num_nodes):
        section = SectionNode(text=f"Section {i}")
        text = TextNode(text=f"Text in Section {i}")
        section.add_child(text)
        root.add_child(section)
        texts.append(text)

    tables: list[TreeNode] = []
    for i in range(table_num_nodes):
        df = pd.DataFrame(
            {
                "Name": [f"Name {i}", f"Name {i + 1}"],
                "Age": [i, i + 1],
                "City": [f"City {i}", f"City {i + 1}"],
            }
        )
        table = TableNode(df=df, description=f"Table {i}")
        row1 = TableRowNode(header=df.columns.tolist(), content=df.iloc[0].tolist())
        row2 = TableRowNode(header=df.columns.tolist(), content=df.iloc[1].tolist())
        table.add_children(row1, row2)
        tables.append(table)

    lists: list[TreeNode] = []
    for i in range(list_num_nodes):
        list_node = ListNode()
        list_item = ListItemNode(text=f"List item {i}-{0}")
        list_node.add_child(list_item)
        list_item = ListItemNode(text=f"List item {i}-{1}")
        list_node.add_child(list_item)
        lists.append(list_node)

    nodes: list[TreeNode] = texts + tables + lists
    root.add_children(*nodes)
    root.update_references()

    original_tree_len = len(list(root.flatten()))

    serialization = root.to_tree_serialization()
    assert serialization

    deserialized = DocumentRootNode(id_=root.id_)
    deserialized.from_tree_serialization(serialization)
    deserialized_tree_len = len(list(deserialized.flatten()))

    assert deserialized_tree_len == original_tree_len


@pytest.mark.parametrize("num_nodes", [1000, 10000, 100000])
def test_document_node_v2_tree_serialization(num_nodes: int) -> None:
    root = DocumentRootNodeV2()
    num_nodes_by_type = num_nodes // 3
    text_num_nodes, table_num_nodes, list_num_nodes = (
        num_nodes_by_type,
        num_nodes_by_type,
        num_nodes_by_type,
    )

    texts: list[TreeNode] = []
    for i in range(text_num_nodes):
        section = SectionNode(text=f"Section {i}")
        text = TextNode(text=f"Text in Section {i}")
        section.add_child(text)
        root.add_child(section)
        texts.append(text)

    tables: list[TreeNode] = []
    for i in range(table_num_nodes):
        df = pd.DataFrame(
            {
                "Name": [f"Name {i}", f"Name {i + 1}"],
                "Age": [i, i + 1],
                "City": [f"City {i}", f"City {i + 1}"],
            }
        )
        table = TableNode(df=df, description=f"Table {i}")
        row1 = TableRowNode(header=df.columns.tolist(), content=df.iloc[0].tolist())
        row2 = TableRowNode(header=df.columns.tolist(), content=df.iloc[1].tolist())
        table.add_children(row1, row2)
        tables.append(table)

    lists: list[TreeNode] = []
    for i in range(list_num_nodes):
        list_node = ListNode()
        list_item = ListItemNode(text=f"List item {i}-{0}")
        list_node.add_child(list_item)
        list_item = ListItemNode(text=f"List item {i}-{1}")
        list_node.add_child(list_item)
        lists.append(list_node)

    nodes: list[TreeNode] = texts + tables + lists
    root.add_children(*nodes)
    root.update_references()

    original_tree_len = len(list(root.flatten()))

    serialization = root.to_tree_serialization()
    assert serialization

    deserialized = DocumentRootNodeV2(id_=root.id_)
    deserialized.from_tree_serialization(serialization)
    deserialized_tree_len = len(list(deserialized.flatten()))

    assert deserialized_tree_len == original_tree_len


def test_combine_trees() -> None:
    root1 = DocumentRootNode()
    section1 = SectionNode(text="Test")
    text1 = TextNode(text="Content")

    root1.add_child(section1)
    section1.add_child(text1)

    root2 = DocumentRootNode()
    section2 = SectionNode(text="Test2")
    text2 = TextNode(text="Content2")
    text3 = TextNode(text="Content3")

    root2.add_child(section2)
    section2.add_child(text2)
    section2.add_child(text3)

    root3 = DocumentRootNode()
    section3 = SectionNode(text="Test3")
    section4 = SectionNode(text="Test4")
    text4 = TextNode(text="Content4")

    root3.add_child(section3)
    root3.add_child(section4)
    section3.add_child(text4)

    combined = combine_trees(root1, root2, root3)
    assert len(combined.children) == 4

    # Check first level
    assert combined.children[0].text == "Test"
    assert combined.children[1].text == "Test2"
    assert combined.children[2].text == "Test3"
    assert combined.children[3].text == "Test4"

    # Check that original nodes are not modified
    assert root1.children[0].text == "Test"

    # Check rest of content
    tree_as_list = list(combined.flatten())
    assert len(tree_as_list) == 9
    assert combined.id_ == root1.id_
    assert all(node.root_id == combined.id_ for node in tree_as_list[1:])


def test_prune_trees() -> None:
    # Case 1. Empty section
    root: TreeNode = DocumentRootNode()
    section = SectionNode(text="Test Section")
    root.add_child(section)
    assert root.children[0] == section
    root = root.prune()
    assert root is None

    # Case 2. Non-empty section
    root: TreeNode = DocumentRootNode()
    text = TextNode(text="Sample text")
    root.add_child(section)
    section.add_child(text)
    assert root.children[0] == section
    assert section.children[0] == text
    root = root.prune()
    assert root is not None
    assert len(root.children) == 1
    assert len(section.children) == 1

    # Case 3. Empty document
    root: TreeNode = DocumentRootNode()
    root = root.prune()
    assert root is None


@pytest.mark.parametrize(
    ("metadata_mode", "metadata", "expected"),
    [
        (
            TreeMetadataMode.ALL,
            {"key1": "value1", "key2": "value2"},
            "key1: value1\nkey2: value2",
        ),
        (
            MetadataMode.LLM,
            {"key1": "value1", "key2": "value2"},
            "key1: value1\nkey2: value2",
        ),
        ("embed", {"key1": "value1", "key2": "value2"}, "key1: value1\nkey2: value2"),
        (TreeMetadataMode.NONE, {"key1": "value1", "key2": "value2"}, ""),
    ],
)
def test_get_metadata_str(
    metadata_mode: TreeMetadataMode | MetadataMode | str,
    metadata: dict[str, str],
    expected: str,
) -> None:
    node = TextNode(text="", extra_info=metadata)
    result = node.get_metadata_str(metadata_mode)
    assert result == expected


@pytest.mark.parametrize(
    ("metadata_mode", "expected"),
    [
        (TreeMetadataMode.ALL, "Content for mode: all"),
        (MetadataMode.LLM, "Content for mode: llm"),
        ("rag", "Content for mode: rag"),
        (TreeMetadataMode.NONE, "Content for mode: none"),
    ],
)
def test_get_content(
    metadata_mode: TreeMetadataMode | MetadataMode | str, expected: str
) -> None:
    node = TextNode(text=expected)
    result = node.get_content(metadata_mode)
    assert result == expected


@pytest.mark.parametrize(
    ("instance", "real_class", "expected"),
    [
        (PartialNode(type=TextNode.get_type()), TextNode, True),
        (PartialNode(type=TextNode.get_type()), SectionNode, False),
        (PartialNode(type=""), None, False),
        (DocumentRootNode(), DocumentRootNode, True),
        (DocumentRootNode(), SectionNode, False),
        (DocumentRootNode(), None, False),
    ],
)
def test_isinstance(instance: Any, real_class: Any, expected: bool) -> None:
    result = instance.isinstance(real_class)
    assert result == expected


def test_add_children() -> None:
    root = DocumentRootNode()

    # Add first section with text
    section1 = SectionNode(text="Introduction")
    root.add_child(section1)
    text1 = TextNode(text="This is the introduction.")
    section1.add_child(text1)

    # Add second section with text
    section2 = SectionNode(text="Body")
    root.add_child(section2)
    text2 = TextNode(text="This is the body text.")
    section2.add_child(text2)

    # Add a subsection with text
    subsection = SectionNode(text="Subsection")
    section2.add_child(subsection)
    subtext = TextNode(text="This is the subsection text.")
    subsection.add_child(subtext)

    # Add a table with rows
    df = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": [30, 25],
            "City": ["New York", "San Francisco"],
        }
    )
    table = TableNode(df=df, description="Sample table")
    section2.add_child(table)
    row1 = TableRowNode(header=df.columns.tolist(), content=df.iloc[0].tolist())
    row2 = TableRowNode(header=df.columns.tolist(), content=df.iloc[1].tolist())
    table.add_child(row1)
    table.add_child(row2)

    # Add a list with items
    list_node = ListNode()
    root.add_child(list_node)
    list_item1 = ListItemNode(text="First item")
    list_item2 = ListItemNode(text="Second item")
    list_node.add_child(list_item1)
    list_node.add_child(list_item2)

    # Add nested list items
    nested_list_item = ListItemNode(text="Nested item")
    list_item2.add_child(nested_list_item)

    # Root update references
    root.update_references()

    # Assertions to check the tree structure
    assert len(root.children) == 3
    assert root.children[0] == section1
    assert root.children[1] == section2
    assert root.children[2] == list_node

    assert len(section1.children) == 1
    assert section1.children[0] == text1

    assert len(section2.children) == 3
    assert section2.children[0] == text2
    assert section2.children[1] == subsection
    assert section2.children[2] == table

    assert len(subsection.children) == 1
    assert subsection.children[0] == subtext

    assert len(table.children) == 2
    assert table.children[0] == row1
    assert table.children[1] == row2

    assert len(list_node.children) == 2
    assert list_node.children[0] == list_item1
    assert list_node.children[1] == list_item2

    assert len(list_item2.children) == 1
    assert list_item2.children[0] == nested_list_item

    # Check parents
    assert section1.parent == root
    assert section2.parent == root
    assert text1.parent == section1
    assert text2.parent == section2
    assert subsection.parent == section2
    assert subtext.parent == subsection
    assert table.parent == section2
    assert row1.parent == table
    assert row2.parent == table
    assert list_node.parent == root
    assert list_item1.parent == list_node
    assert list_item2.parent == list_node
    assert nested_list_item.parent == list_item2

    # Check root id
    assert section1.root_id == root.id_
    assert section2.root_id == root.id_
    assert text1.root_id == root.id_
    assert text2.root_id == root.id_
    assert subsection.root_id == root.id_
    assert subtext.root_id == root.id_
    assert table.root_id == root.id_
    assert row1.root_id == root.id_
    assert row2.root_id == root.id_
    assert list_node.root_id == root.id_
    assert list_item1.root_id == root.id_
    assert list_item2.root_id == root.id_
    assert nested_list_item.root_id == root.id_

    # Check depth
    assert root.depth == 0
    assert section1.depth == 1
    assert section2.depth == 1
    assert text1.depth == 2
    assert text2.depth == 2
    assert subsection.depth == 2
    assert subtext.depth == 3
    assert table.depth == 2
    assert row1.depth == 3
    assert row2.depth == 3
    assert list_node.depth == 1
    assert list_item1.depth == 2
    assert list_item2.depth == 2
    assert nested_list_item.depth == 3

    # Check order index
    assert root.idx == 0
    assert section1.idx == 0
    assert section2.idx == 1
    assert text1.idx == 0
    assert text2.idx == 0
    assert subsection.idx == 1
    assert subtext.idx == 0
    assert table.idx == 2
    assert row1.idx == 0
    assert row2.idx == 1
    assert list_node.idx == 2
    assert list_item1.idx == 0
    assert list_item2.idx == 1
    assert nested_list_item.idx == 0

    # Check height
    assert root.height == 3
    assert section1.height == 1
    assert section2.height == 2
    assert text1.height == 0
    assert text2.height == 0
    assert subsection.height == 1
    assert subtext.height == 0
    assert table.height == 1
    assert row1.height == 0
    assert row2.height == 0
    assert list_node.height == 2
    assert list_item1.height == 0
    assert list_item2.height == 1
    assert nested_list_item.height == 0

    # Check absolute index
    for i, node in enumerate(root.flatten()):
        assert node.abs_idx == i


def test_insert_node() -> None:
    root = DocumentRootNode()
    section1 = SectionNode(text="Introduction")
    root.add_child(section1)
    text1 = TextNode(text="This is the introduction.")
    section1.add_child(text1)
    section2 = SectionNode(text="Body")
    root.add_child(section2)
    text2 = TextNode(text="This is the body text.")
    section2.add_child(text2)
    subsection = SectionNode(text="Subsection")
    section2.add_child(subsection)
    subtext = TextNode(text="This is the subsection text.")
    subsection.add_child(subtext)
    root.update_references()

    # Case 1. Insert element at the end
    new_text = TextNode(text="This is the new text.")
    root.insert_child(2, new_text, update_references=True)
    sorted_nodes = list(root.flatten())

    assert len(root.children) == 3
    assert root.children[2] == new_text

    assert root.height == 3
    assert new_text.height == 0
    assert new_text.abs_idx == sorted_nodes.index(new_text)

    # Case 2. Insert element at the beginning
    new_section = SectionNode(text="New Section")
    root.insert_child(0, new_section)
    root.update_references()
    sorted_nodes = list(root.flatten())

    assert len(root.children) == 4
    assert root.children[0] == new_section

    assert root.height == 3
    assert new_section.height == 0
    assert new_section.abs_idx == sorted_nodes.index(new_section)
    for i, child in enumerate(root.children):
        assert child.idx == i

    # Case 3. Insert element in the middle
    new_subsection = SectionNode(text="New Subsection")
    root.insert_child(2, new_subsection, update_references=True)
    sorted_nodes = list(root.flatten())

    assert len(root.children) == 5
    assert root.children[2] == new_subsection

    assert root.height == 3
    assert new_subsection.height == 0
    assert new_subsection.abs_idx == sorted_nodes.index(new_subsection)
    for i, child in enumerate(root.children):
        assert child.idx == i

    # Case 4. Negative index
    new_text = TextNode(text="This is the new text.")
    root.insert_child(-1, new_text)
    root.update_references()
    sorted_nodes = list(root.flatten())

    assert len(root.children) == 6
    assert root.children[-2] == new_text

    assert root.height == 3
    assert new_text.height == 0
    assert new_text.abs_idx == sorted_nodes.index(new_text)
    for i, child in enumerate(root.children):
        assert child.idx == i

    # Case 5.1 Index out of bounds (negative)
    new_text = TextNode(text="This is the new text.")
    with pytest.raises(IndexError):
        root.insert_child(-100, new_text)

    # Case 5.2 Index out of bounds (positive)
    new_text = TextNode(text="This is the new text.")
    with pytest.raises(IndexError):
        root.insert_child(100, new_text)


def test_partial_node() -> None:
    def compare_nodes(node1: Any, node2: Any) -> None:
        assert node1.id_ == node2.id_
        assert node1.metadata == node2.metadata
        assert node1.type == node2.get_type()
        assert node1.idx == node2.idx
        assert node1.depth == node2.depth
        assert node1.parent_id == node2.parent_id
        assert node1.root_id == node2.root_id
        assert node1.hash == node2.hash
        assert node1.height == node2.height
        assert node1.abs_idx == node2.abs_idx
        assert len(node1.children) == len(node2.children)
        for child1, child2 in zip(node1.children, node2.children, strict=False):
            compare_nodes(child1, child2)

    # Case 1. Create a partial node from a tree node
    root = DocumentRootNode()
    section1 = SectionNode(text="Introduction")
    root.add_child(section1)
    text1 = TextNode(text="This is the introduction.")
    section1.add_child(text1)
    section2 = SectionNode(text="Body")
    root.add_child(section2)

    partial_node = PartialNode.from_node(root)
    compare_nodes(partial_node, root)

    # Case 2. Create a partial node from a dictionary
    dict = partial_node.dict()

    partial_node2 = PartialNode.from_partial_dict(dict)
    compare_nodes(partial_node2, root)
