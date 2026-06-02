import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import BaseNode

from private_gpt.components.readers.nodes.utils import metadata_dict_to_tree_node


@patch("llama_index.core.vector_stores.utils.metadata_dict_to_node")
def test_legacy_node_type(mock_metadata_dict_to_node: MagicMock) -> None:
    mock_node = MagicMock(spec=BaseNode)
    mock_metadata_dict_to_node.return_value = mock_node

    metadata: dict[str, Any] = {
        "_node_content": '{"key": "value"}',
        "_node_type": "LegacyNodeType",
    }

    node = metadata_dict_to_tree_node(metadata)
    assert node == mock_node
    mock_metadata_dict_to_node.assert_called_once_with(metadata)


def test_missing_node_content() -> None:
    metadata: dict[str, Any] = {"_node_type": "DocumentRoot-v1"}
    with pytest.raises(
        ValueError, match=re.escape("Node content not found in metadata dict.")
    ):
        metadata_dict_to_tree_node(metadata)


def test_missing_node_type() -> None:
    metadata: dict[str, Any] = {"_node_content": '{"key": "value"}'}
    with pytest.raises(
        ValueError, match=re.escape("Node type not found in metadata dict.")
    ):
        metadata_dict_to_tree_node(metadata)


def test_unknown_node_type() -> None:
    metadata: dict[str, Any] = {
        "_node_content": '{"key": "value"}',
        "_node_type": "UnknownNode-v1",
    }
    with pytest.raises(ValueError, match="Unknown node type: UnknownNode-v1"):
        metadata_dict_to_tree_node(metadata)


def test_unknown_version() -> None:
    metadata: dict[str, Any] = {
        "_node_content": '{"key": "value"}',
        "_node_type": "DocumentRoot-v2",
    }
    with pytest.raises(ValueError, match="Unknown node type: DocumentRoot-v2"):
        metadata_dict_to_tree_node(metadata)
