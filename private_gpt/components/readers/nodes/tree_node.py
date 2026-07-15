import builtins
import enum
import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from hashlib import sha256
from typing import Any, Optional, Self, Union, overload

from llama_index.core.base.llms.types import BaseContentBlock, TextBlock
from llama_index.core.schema import (
    DEFAULT_METADATA_TMPL,
    BaseNode,
    MetadataMode,
    RelatedNodeInfo,
)
from pydantic import Field, model_serializer
from pydantic_core.core_schema import SerializationInfo, SerializerFunctionWrapHandler

from private_gpt.components.ingest.metadata_helper import MetadataNode

CURRENT_VERSION = "v1"


class TreeMetadataMode(enum.StrEnum):
    ALL = "all"
    EMBED = "embed"
    LLM = "llm"
    NONE = "none"
    RAG = "rag"
    USER = "user"

    @classmethod
    def from_enum_or_str(cls, value: str) -> "TreeMetadataMode":
        if isinstance(value, cls):
            return value
        if isinstance(value, Enum):
            return cls.from_str(value.value)
        return cls.from_str(value)

    @classmethod
    def from_str(cls, value: str) -> "TreeMetadataMode":
        try:
            return TreeMetadataMode(value)
        except ValueError as e:
            raise ValueError(f"Invalid mode: {value}") from e


class TreeNode(BaseNode, ABC):
    """Abstract base class for tree nodes."""

    parent: Optional["TreeNode"] = Field(default=None, description="Parent node.")
    children: list["TreeNode"] = Field(
        default_factory=list, description="List of child nodes."
    )

    parent_id: str | None = None
    root_id: str | None = None
    depth: int = Field(default=0, description="Depth of the node in the tree.")
    height: int = Field(default=0, description="Height of the node in the tree.")
    idx: int = Field(
        default=0,
        description="Index of the node in the list of children of the parent node.",
    )
    abs_idx: int = Field(
        default=0,
        description="Absolute index of the node in the list of all nodes.",
    )

    metadata_template: str = Field(
        default=DEFAULT_METADATA_TMPL,
        description=(
            "Template for how metadata is formatted, with {key} and "
            "{value} placeholders."
        ),
    )
    metadata_separator: str = Field(
        default="\n",
        description="Separator between metadata fields when converting to string.",
    )

    @property
    def token_count(self) -> int:
        return self.metadata.get(MetadataNode.TOKEN_COUNT) or 0

    def add_children(
        self, *children: "TreeNode", update_references: bool = False
    ) -> None:
        """Add multiple children nodes."""
        for child in children:
            if update_references:
                child.depth = self.depth + 1
                child.idx = len(self.children)
                child._update_height(move_upwards=False)

            # Avoid to update references to avoid multiple recalculations
            self.add_child(child)

        if update_references:
            self._update_height()
            self._recalculate_relative_indices(self.idx)
            self._recalculate_absolute_indices()

    def add_child(self, child: "TreeNode", update_references: bool = False) -> None:
        """Add a child node."""
        self.insert_child(
            len(self.children), child, update_references=update_references
        )

    def insert_child(
        self, index: int, child: "TreeNode", update_references: bool = False
    ) -> None:
        """Insert a child node at a specific index."""
        if index < -len(self.children) or index > len(self.children):
            raise IndexError("Index out of bounds")

        if index < 0:
            index += len(self.children)

        # Update child attributes
        child.parent = self
        child.parent_id = self.id_
        child.root_id = self.root_id or self.id_

        if update_references:
            child.depth = self.depth + 1
            child.idx = index

        # Insert child
        if index == len(self.children):
            self.children.append(child)
        else:
            self.children.insert(index, child)

        # Update rest of the tree values
        if update_references:
            self._update_height()
            self._recalculate_relative_indices(start_idx=index)
            self._recalculate_absolute_indices()

    def _update_height(self, move_upwards: bool = True) -> None:
        """Update the height of the node and its ancestors."""
        current: TreeNode | None = self
        while current:
            new_height = (
                max((child.height for child in current.children), default=-1) + 1
            )
            if new_height == current.height:
                # If height hasn't changed, ancestors won't change either
                break
            current.height = new_height
            if not move_upwards:
                break
            current = current.parent

    def _recalculate_absolute_indices(self) -> None:
        """Recalculate absolute indices for the entire tree."""
        root = self
        while root.parent is not None:
            root = root.parent

        root.abs_idx = 0
        for i, node in enumerate(root.flatten()):
            node.abs_idx = i

    def _recalculate_relative_indices(self, start_idx: int = 0) -> None:
        """Recalculate indices for the parent's children."""
        for i, sibling in enumerate(self.children[start_idx:], start=start_idx):
            sibling.idx = i

    def update_references(self) -> None:
        """Refresh all tree-related values in a single pass.

        Updates:
        - depth: distance from root
        - height: length of longest path to leaf
        - idx: position among siblings
        - abs_idx: position in entire tree
        - parent/child relationships
        - root_id references
        """
        root = self
        while root.parent is not None:
            root = root.parent

        # Initialize counters
        abs_idx_counter = 0

        def update_node_recursive(
            node: "TreeNode", current_depth: int, parent: Optional["TreeNode"] = None
        ) -> int:
            nonlocal abs_idx_counter

            # Update basic node properties
            node.depth = current_depth
            node.parent = parent
            node.parent_id = parent.id_ if parent else None
            node.root_id = root.id_
            node.abs_idx = abs_idx_counter
            abs_idx_counter += 1

            # Update child indices
            for idx, child in enumerate(node.children):
                child.idx = idx

            # Recursively process children and calculate height
            max_child_height = -1
            for child in node.children:
                child_height = update_node_recursive(child, current_depth + 1, node)
                max_child_height = max(max_child_height, child_height)

            # Set node height based on children
            node.height = max_child_height + 1
            return node.height

        # Start the recursive update from root
        update_node_recursive(root, 0, None)

    def flatten(self) -> Iterator["TreeNode"]:
        """Perform a DFS to flatten the tree."""
        yield self
        for child in self.children:
            yield from child.flatten()

    def late_flatten(self) -> Iterator["TreeNode"]:
        """Perform a DFS to flatten the tree."""
        for child in self.children:
            yield from child.late_flatten()
        yield self

    @classmethod
    def version(cls) -> str:
        return CURRENT_VERSION

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__ + "-" + cls.version()

    @classmethod
    def get_type(cls) -> str:
        return cls.class_name()

    def isinstance(
        self, cls: builtins.type["TreeNode"] | tuple[builtins.type["TreeNode"], ...]
    ) -> bool:
        return bool(cls and isinstance(self, cls))

    @property
    def hash(self) -> str:
        doc_identity = f"{self.get_type()}-{self.get_content(TreeMetadataMode.NONE)}"
        return str(sha256(doc_identity.encode("utf-8", "surrogatepass")).hexdigest())

    def __hash__(self) -> int:
        return hash(self.hash)

    def as_related_node_info(self) -> RelatedNodeInfo:
        """Get node as RelatedNodeInfo."""
        return RelatedNodeInfo(
            node_id=self.node_id,
            metadata=self.metadata,
            hash=self.hash,
        )

    def __str__(self) -> str:
        from textwrap import shorten

        content = shorten(self.get_content(TreeMetadataMode.RAG), 50)
        return f"{self.get_type()}({self.id_})" + (f": {content}" if content else "")

    def __repr__(self) -> str:
        return self.__str__()

    @model_serializer(mode="wrap")
    def custom_model_dump(
        self,
        handler: SerializerFunctionWrapHandler,
        info: SerializationInfo,
    ) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)

        # Add relevant metadata
        data["class_name"] = self.class_name()
        data["version"] = self.version()

        # Add metadata
        if self.root_id:
            data["root_id"] = self.root_id

        return data

    def model_dump(
        self,
        *,
        include_parent: bool = False,
        include_children: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize to dictionary, avoiding circular references."""
        exclude: set[str] = set(kwargs.pop("exclude", []))
        if not include_parent:
            exclude.add("parent")
        # Exclude always children to prevent circular references
        exclude.add("children")

        serialized = super().model_dump(exclude=exclude, **kwargs)

        # Handle children serialization separately to avoid recursion issues
        if include_children:
            serialized["children"] = [
                child.dict(
                    include_parent=include_parent, include_children=include_children
                )
                for child in self.children or []
            ]
        return serialized

    def model_dump_json(
        self,
        *,
        indent: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Serialize to JSON string, avoiding circular references."""
        data = self.model_dump(**kwargs)
        return str(json.dumps(data, indent=indent))

    @classmethod
    def from_dict(cls, data: builtins.dict[str, Any], **kwargs: Any) -> Self:
        if data["class_name"] != cls.class_name():
            raise ValueError(
                f"Expected class {cls.class_name()}, got {data['class_name']}"
            )
        return super().from_dict(data, **kwargs)

    @overload
    def get_metadata_str(self, mode: TreeMetadataMode = TreeMetadataMode.ALL) -> str:
        ...

    @overload
    def get_metadata_str(self, mode: MetadataMode = MetadataMode.ALL) -> str:
        ...

    @overload
    def get_metadata_str(self, mode: str = "all") -> str:
        ...

    def get_metadata_str(
        self, mode: TreeMetadataMode | MetadataMode | str = TreeMetadataMode.ALL
    ) -> str:
        metadata_mode: TreeMetadataMode = TreeMetadataMode.from_enum_or_str(mode)
        return self.get_metadata_str_internal(metadata_mode)

    def get_metadata_str_internal(self, metadata_mode: TreeMetadataMode) -> str:
        if metadata_mode == TreeMetadataMode.NONE:
            return ""

        usable_metadata_keys = set(self.metadata.keys())
        if metadata_mode == TreeMetadataMode.LLM:
            usable_metadata_keys -= set(self.excluded_llm_metadata_keys)
        elif metadata_mode == TreeMetadataMode.EMBED:
            usable_metadata_keys -= set(self.excluded_embed_metadata_keys)
        elif (
            metadata_mode == TreeMetadataMode.RAG
            or metadata_mode == TreeMetadataMode.USER
        ):
            usable_metadata_keys -= set(self.excluded_embed_metadata_keys)
            usable_metadata_keys -= set(self.excluded_llm_metadata_keys)

        return self.metadata_separator.join(
            [
                self.metadata_template.format(key=key, value=str(value))
                for key, value in self.metadata.items()
                if key in usable_metadata_keys
            ]
        )

    @overload
    def get_content(
        self, metadata_mode: TreeMetadataMode = TreeMetadataMode.ALL
    ) -> str:
        ...

    @overload
    def get_content(self, metadata_mode: MetadataMode = MetadataMode.ALL) -> str:
        ...

    @overload
    def get_content(self, metadata_mode: str = "all") -> str:
        ...

    def get_content(
        self,
        metadata_mode: TreeMetadataMode | MetadataMode | str = TreeMetadataMode.ALL,
    ) -> str:
        tree_mode: TreeMetadataMode = TreeMetadataMode.from_enum_or_str(metadata_mode)
        return self.get_content_internal(metadata_mode=tree_mode)

    @abstractmethod
    def get_content_internal(self, metadata_mode: TreeMetadataMode) -> str:
        pass

    @overload
    def get_content_blocks(
        self, metadata_mode: TreeMetadataMode = TreeMetadataMode.ALL
    ) -> list[BaseContentBlock]:
        ...

    @overload
    def get_content_blocks(
        self, metadata_mode: MetadataMode = MetadataMode.ALL
    ) -> list[BaseContentBlock]:
        ...

    @overload
    def get_content_blocks(self, metadata_mode: str = "all") -> list[BaseContentBlock]:
        ...

    def get_content_blocks(
        self,
        metadata_mode: TreeMetadataMode | MetadataMode | str = TreeMetadataMode.ALL,
    ) -> list[BaseContentBlock]:
        """Get content blocks for the node."""
        blocks: list[BaseContentBlock] = [
            TextBlock(text=self.get_content(metadata_mode))
        ]
        return blocks

    def print_tree(self, indent: int = 2) -> None:
        """Print the tree."""
        from private_gpt.components.readers.nodes.node_print import TreePrinter

        TreePrinter.print(
            self,
            indent_size=indent,
        )

    def find_self_or_child_by_id(self, node_id: str) -> Optional["TreeNode"]:
        """Recursively find a child node by ID."""
        if self.id_ == node_id:
            return self
        for child in self.children:
            if child.id_ == node_id:
                return child
            found: TreeNode | None = child.find_self_or_child_by_id(node_id)
            if found:
                return found
        return None

    def prune(
        self,
        metadata_mode: TreeMetadataMode | MetadataMode | str = TreeMetadataMode.LLM,
    ) -> Union["TreeNode", None]:
        """Prune the tree where necessary."""
        new_children = []
        for _, child in enumerate(self.children):
            reduced = child.prune(metadata_mode)
            if reduced:
                new_children.append(reduced)

        # Clean last memory reference to children
        self.children.clear()
        for child in new_children:
            self.add_child(child, update_references=False)

        # By default, prune all subtrees that don't contain content
        content = self.get_content(metadata_mode).strip()
        if not content:
            return None

        # Return the non-pruned node
        return self

    def get_sum_token_count(self) -> int:
        """Get the sum of token counts of all children."""
        return self.token_count + sum(
            child.get_sum_token_count() for child in self.children
        )

    @classmethod
    def clone_tree(
        cls,
        node: "TreeNode",
        **kwargs: Any,
    ) -> "TreeNode":
        """Clones the tree structure starting from the given node."""
        return cls.from_dict(
            data=node.dict(
                include_parent=False,
                include_children=False,
                **kwargs,
            )
        )

    @classmethod
    def rebuild_tree(
        cls, nodes: list["TreeNode"], root_node: Union["TreeNode", None] = None
    ) -> list["TreeNode"]:
        """Rebuilds the tree structure."""
        # Create a lookup dictionary for all nodes
        nodes_by_id: dict[str, TreeNode] = {node.id_: node for node in nodes}
        full_nodes_by_id: dict[str, TreeNode] = (
            {node.id_: node for node in root_node.flatten()} if root_node else {}
        )

        # Group nodes by their parent_id
        missing_parent_ids: set[str] = set()
        nodes_by_parent: dict[str | None, list[TreeNode]] = {}
        for node in nodes:
            if node.parent_id not in nodes_by_parent:
                nodes_by_parent[node.parent_id] = []
            nodes_by_parent[node.parent_id].append(node)
            if node.parent_id and node.parent_id not in nodes_by_id:
                missing_parent_ids.add(node.parent_id)

        # Create all missing parent nodes as PartialTreeNode instances
        for parent_id in missing_parent_ids:
            from private_gpt.components.readers.nodes.partial_node import PartialNode

            original_node = full_nodes_by_id.get(parent_id)
            if original_node:
                nodes_by_id[parent_id] = PartialNode.from_node(
                    original_node.__class__.clone_tree(original_node)
                )
            else:
                # If the parent node is missing, create a placeholder node
                nodes_by_id[parent_id] = PartialNode(id_=parent_id, type="Unknown")

        # Rebuild the tree structure
        for p_id, nodes in nodes_by_parent.items():
            if p_id is None:
                continue
            parent = nodes_by_id[p_id]
            parent.children.clear()
            parent.add_children(*sorted(nodes, key=lambda n: n.abs_idx))

        # Create a list of root nodes
        roots: dict[str, TreeNode] = {}
        for node in nodes_by_id.values():
            if node.parent_id is None:
                # We have a full root node
                roots[node.id_] = node
            elif nodes_by_id.get(node.parent_id) is None:
                # We have a partial root node
                roots[node.parent_id] = node

        return list(roots.values())
