import json
from typing import Any

from llama_index.core.schema import BaseNode

from private_gpt.components.readers.nodes import DiffNode
from private_gpt.components.readers.nodes.chunk_node import ChunkNode
from private_gpt.components.readers.nodes.document_node import DocumentRootNode
from private_gpt.components.readers.nodes.fragment_node import FragmentRootNode
from private_gpt.components.readers.nodes.frozen_node import FrozenNode
from private_gpt.components.readers.nodes.image_node import ImageNode
from private_gpt.components.readers.nodes.list_node import ListItemNode, ListNode
from private_gpt.components.readers.nodes.partial_node import PartialNode
from private_gpt.components.readers.nodes.section_node import SectionNode
from private_gpt.components.readers.nodes.table_node import TableNode, TableRowNode
from private_gpt.components.readers.nodes.text_node import TextNode
from private_gpt.components.readers.nodes.tree_node import TreeNode
from private_gpt.components.readers.nodes.v2.document_node_v2 import DocumentRootNodeV2


def dict_to_tree_node(
    version: str, node_type: str, node_dict: dict[str, Any]
) -> TreeNode:
    match version:
        case "v1":
            match node_type:
                case DocumentRootNode.__name__:
                    return DocumentRootNode.from_dict(node_dict)
                case SectionNode.__name__:
                    return SectionNode.from_dict(node_dict)
                case TableNode.__name__:
                    return TableNode.from_dict(node_dict)
                case TableRowNode.__name__:
                    return TableRowNode.from_dict(node_dict)
                case TextNode.__name__:
                    return TextNode.from_dict(node_dict)
                case ListNode.__name__:
                    return ListNode.from_dict(node_dict)
                case ListItemNode.__name__:
                    return ListItemNode.from_dict(node_dict)
                case ChunkNode.__name__:
                    return ChunkNode.from_dict(node_dict)
                case ImageNode.__name__:
                    return ImageNode.from_dict(node_dict)
                case FragmentRootNode.__name__:
                    return FragmentRootNode.from_dict(node_dict)
                case PartialNode.__name__:
                    return PartialNode.from_dict(node_dict)
                case DiffNode.__name__:
                    return DiffNode.from_dict(node_dict)
                case FrozenNode.__name__:
                    return FrozenNode.from_dict(node_dict)
                # Any other missing node types
        case "v2":
            match node_type:
                case DocumentRootNodeV2.__name__:
                    return DocumentRootNodeV2.from_dict(node_dict)

    raise ValueError(f"Unknown node type: {node_type}-{version}")


def json_to_tree_node(version: str, node_type: str, node_json: str) -> TreeNode:
    node_dict = json.loads(node_json)
    return dict_to_tree_node(version, node_type, node_dict)


def metadata_dict_to_tree_node(metadata: dict[str, Any]) -> BaseNode:
    """Common logic for loading Node data from metadata dict."""
    node_json: str | None = metadata.get("_node_content")
    node_type: str | None = metadata.get("_node_type")

    if node_json is None:
        raise ValueError("Node content not found in metadata dict.")
    if node_type is None:
        raise ValueError("Node type not found in metadata dict.")

    if "-" not in node_type:
        # This is a legacy node type
        from llama_index.core.vector_stores.utils import metadata_dict_to_node

        return metadata_dict_to_node(metadata)

    node_type, version = node_type.split("-")
    return json_to_tree_node(version, node_type, node_json)


def combine_trees(root: TreeNode, *trees: TreeNode) -> TreeNode:
    def update_root_id(node: TreeNode, new_root_id: str) -> None:
        node.root_id = new_root_id
        for child in node.children:
            update_root_id(child, new_root_id)

    def combine_trees(root1: TreeNode, root2: TreeNode) -> TreeNode:
        for child in root2.children:
            update_root_id(child, root1.id_)

        # Move children from root2 to root1
        root1.children.extend(root2.children)
        root2.children = []

        return root1

    root_copy = root.model_copy()
    for tree in trees:
        tree_copy = tree.model_copy()
        root_copy = combine_trees(root_copy, tree_copy)

    return root_copy
