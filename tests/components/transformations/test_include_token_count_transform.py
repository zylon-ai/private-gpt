from collections.abc import Iterator

import pandas as pd
import pytest

from private_gpt.components.ingest.metadata_helper import MetadataNode
from private_gpt.components.ingest.transformations.include_token_count_to_nodes_transform import (
    IncludeTokenCountIntoNodesTransform,
)
from private_gpt.components.readers.nodes.diff_node import DiffNode
from private_gpt.components.readers.nodes.document_node import DocumentRootNode
from private_gpt.components.readers.nodes.image_node import ImageNode
from private_gpt.components.readers.nodes.list_node import ListItemNode, ListNode
from private_gpt.components.readers.nodes.partial_node import PartialNode
from private_gpt.components.readers.nodes.section_node import SectionNode
from private_gpt.components.readers.nodes.table_node import TableNode, TableRowNode
from private_gpt.components.readers.nodes.text_node import TextNode
from private_gpt.components.readers.nodes.tree_node import TreeNode


def mock_metadata() -> dict[str, str]:
    return {
        "key1": "value1",
        "key2": "value2",
    }


def mock_children() -> Iterator[TreeNode]:
    yield SectionNode(
        children=[
            TextNode(
                text="This is a test",
                extra_info=mock_metadata(),
            ),
            TextNode(
                text="This is a test",
            ),
        ],
    )
    yield TextNode(
        text="This is a test",
        extra_info=mock_metadata(),
    )


@pytest.fixture
def transform() -> IncludeTokenCountIntoNodesTransform:
    return IncludeTokenCountIntoNodesTransform.from_defaults(
        tokenizer=lambda x: x.split()
    )


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (
            DocumentRootNode(
                children=list(mock_children()),
                extra_info=mock_metadata(),
                excluded_llm_metadata_keys=list(mock_metadata().keys()),
            ),
            0,
        ),
        (
            TextNode(
                text="This is a test",
                extra_info=mock_metadata(),
                excluded_llm_metadata_keys=list(mock_metadata().keys()),
            ),
            4,
        ),
        (
            SectionNode(
                text="This is a test",
                children=list(mock_children()),
                extra_info=mock_metadata(),
                excluded_llm_metadata_keys=list(mock_metadata().keys()),
            ),
            4,
        ),
        (
            ImageNode(
                alt_text="This is a test",
                image="#",
                extra_info=mock_metadata(),
                excluded_llm_metadata_keys=list(mock_metadata().keys()),
            ),
            4,
        ),
        (
            ListNode(
                children=list(mock_children()),
                extra_info=mock_metadata(),
                excluded_llm_metadata_keys=list(mock_metadata().keys()),
            ),
            0,
        ),
        (
            ListItemNode(
                text="This is a test",
                extra_info=mock_metadata(),
                excluded_llm_metadata_keys=list(mock_metadata().keys()),
            ),
            4,
        ),
        (
            TableNode(
                df=pd.DataFrame(
                    data={
                        "Column 1": ["Value 1", "Value 2"],
                        "Column 2": ["Value 3", "Value 4"],
                    }
                ),
            ),
            0,
        ),
        (
            TableRowNode(
                header=["Column 1", "Column 2"],
                content=["Value 1", "Value 2"],
                extra_info=mock_metadata(),
                excluded_llm_metadata_keys=list(mock_metadata().keys()),
            ),
            7,
        ),
        (
            PartialNode(type="test", extra_info=mock_metadata()),
            0,
        ),
        (
            DiffNode(
                text="This is a test",
                extra_info=mock_metadata(),
                excluded_llm_metadata_keys=list(mock_metadata().keys()),
            ),
            4,
        ),
    ],
)
def test_token_count(
    transform: IncludeTokenCountIntoNodesTransform,
    node: TreeNode,
    expected: int,
) -> None:
    transformed_nodes = transform([node])
    token_count = transformed_nodes[0].metadata.get(MetadataNode.TOKEN_COUNT.value) or 0
    assert token_count == expected
    if expected:
        assert (
            MetadataNode.TOKEN_COUNT.value
            in transformed_nodes[0].excluded_llm_metadata_keys
        )
        assert (
            MetadataNode.TOKEN_COUNT.value
            in transformed_nodes[0].excluded_embed_metadata_keys
        )
