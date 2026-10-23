import textwrap
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Generic, TypeVar, cast

from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode

T = TypeVar("T", bound=TreeNode)


class NodeColor(Enum):
    HEADER = "\033[95m"  # purple
    TEXT = "\033[94m"  # blue
    TABLE = "\033[92m"  # green
    LIST = "\033[91m"  # red
    ROOT = "\033[93m"  # yellow
    BRANCH = "\033[90m"  # gray
    DEFAULT = "\033[97m"  # white
    END = "\033[0m"  # reset


@dataclass
class TreePrintConfig:
    """Configuration for tree printing."""

    show_metadata: bool = False
    show_num_tokens: bool = True
    max_content_length: int = 50
    show_timestamps: bool = False
    show_content_preview: bool = True
    indent_size: int = 2
    show_node_count: bool = False
    show_types: bool = True
    show_ids: bool = True


class TreePrinter(Generic[T]):
    """Helper class for printing tree structures with advanced formatting."""

    def __init__(self, config: TreePrintConfig | None = None):
        self.config = config or TreePrintConfig()

    def _get_node_color(self, node: Any) -> str:
        """Determine color based on node type name."""
        type_name = type(node).__name__.lower()
        if "root" in type_name:
            return NodeColor.ROOT.value
        elif "section" in type_name or "header" in type_name:
            return NodeColor.HEADER.value
        elif "table" in type_name:
            return NodeColor.TABLE.value
        elif "text" in type_name:
            return NodeColor.TEXT.value
        elif "list" in type_name or "item" in type_name:
            return NodeColor.LIST.value
        return NodeColor.DEFAULT.value

    def _get_node_summary(self, node: Any) -> str:
        """Generate a summary string for a node."""
        parts = []

        # Add node type if configured
        if self.config.show_types:
            parts.append(f"{type(node).__name__}")

        if self.config.show_ids:
            parts.append(f"[{node.id_}]")

        # Handle different node attributes
        if hasattr(node, "content") and node.content:
            preview = textwrap.shorten(
                str(node.content),
                width=self.config.max_content_length,
                placeholder="...",
            )
            parts.append(f"'{preview}'")

        # Special handling for table-like nodes
        if hasattr(node, "df"):
            try:
                rows, cols = node.df.shape
                parts.append(f"[{rows}x{cols} table]")
            except AttributeError:
                pass

        return " ".join(parts)

    def _get_metadata_string(self, node: Any) -> str:
        """Format node metadata if available."""
        if hasattr(node, "metadata") and node.metadata:
            return f"{NodeColor.BRANCH.value}[meta: {node.metadata}]"
        return ""

    def print_tree(
        self,
        node: T,
        level: int = 0,
        is_last: bool = True,
        prefix: str = "",
        parent_prefix: str = "",
    ) -> None:
        """Print a tree structure with advanced formatting.

        Args:
            node: The current node to print
            level: Current depth in the tree
            is_last: Whether this is the last child of its parent
            prefix: Current line prefix
            parent_prefix: Prefix for parent connection lines
        """
        # Branch characters
        branch = "└── " if is_last else "├── "

        # Get node color and info
        color = self._get_node_color(node)
        node_info = self._get_node_summary(node)

        # Build the line
        line_parts = []

        # Add branch with gray color
        if level > 0:
            line_parts.append(f"{NodeColor.BRANCH.value}{parent_prefix}{branch}")

        # Add colored node information
        line_parts.append(f"{color}{node_info}")

        # Add metadata if configured
        if self.config.show_metadata:
            metadata = self._get_metadata_string(node)
            if metadata:
                line_parts.append(metadata)

        # Add number of tokens if configured
        if self.config.show_num_tokens:
            line_parts.append(f"{NodeColor.BRANCH.value}[{node.token_count} tokens]")

        # Add note content preview if configured
        if self.config.show_content_preview:
            content_preview = node.get_content(TreeMetadataMode.NONE)
            if content_preview:
                preview = textwrap.shorten(
                    str(content_preview),
                    width=self.config.max_content_length,
                    placeholder="...",
                )
                line_parts.append(f"{NodeColor.BRANCH.value}[{preview}]")

        # Add node count if configured
        if self.config.show_node_count:
            children_count = len(node.children or [])
            line_parts.append(f"{NodeColor.BRANCH.value}[{children_count} children]")

        # Add timestamp if configured
        if self.config.show_timestamps:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            line_parts.append(f"{NodeColor.BRANCH.value}[{timestamp}]")

        # Print the complete line
        print("".join(line_parts) + NodeColor.END.value)

        # Handle children
        if hasattr(node, "children"):
            children = node.children
            if not children:
                return
            new_parent_prefix = parent_prefix + ("    " if is_last else "│   ")

            for i, child in enumerate(children):
                self.print_tree(
                    node=cast(T, child),
                    level=level + 1,
                    is_last=(i == len(children) - 1),
                    prefix=prefix,
                    parent_prefix=new_parent_prefix,
                )

    @classmethod
    def print(
        cls,
        node: T,
        *,
        indent_size: int = 2,
    ) -> None:
        """Convenient class method for quick printing with custom config.

        Example:
            TreePrinter.print(
                root_node,
                show_metadata=True,
                max_content_length=30,
                show_timestamps=False
            )
        """
        config = TreePrintConfig(
            indent_size=indent_size,
        )
        printer = cls(config)
        printer.print_tree(node)
