from typing import Union

from llama_index.core.schema import MetadataMode

from private_gpt.components.readers.nodes.text_node import TextNode
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode


class DiffProcessor:
    @staticmethod
    def diff(
        ref_text: str,
        other_text: str,
    ) -> str | None:

        if ref_text == other_text:
            return None

        ref_lines = ref_text.splitlines(keepends=True)
        other_lines = other_text.splitlines(keepends=True)

        result = []
        ref_index = 0
        other_index = 0
        skipped_count = 0

        while ref_index < len(ref_lines) and other_index < len(other_lines):
            if ref_lines[ref_index] == other_lines[other_index]:
                if skipped_count > 0:
                    result.append(f"@@ Skipped {skipped_count} lines @@\n")
                    skipped_count = 0

                result.append(ref_lines[ref_index])
                ref_index += 1
                other_index += 1
            else:
                while (
                    ref_index < len(ref_lines)
                    and other_index < len(other_lines)
                    and ref_lines[ref_index] != other_lines[other_index]
                ):
                    ref_index += 1
                    skipped_count += 1

        # Add remaining lines
        while other_index < len(other_lines):
            result.append(other_lines[other_index])
            other_index += 1

        return "".join(result) if result else None


class DiffNode(TextNode):
    """Diff node."""

    @classmethod
    def from_nodes(
        cls,
        ref_node: TreeNode,
        other_node: TreeNode,
        metadata_mode: TreeMetadataMode | MetadataMode | str = TreeMetadataMode.ALL,
    ) -> Union["DiffNode", None]:
        """Create a diff node from a list of nodes."""
        diff = DiffProcessor.diff(
            ref_text=ref_node.get_content(metadata_mode),
            other_text=other_node.get_content(metadata_mode),
        )
        return cls(text=diff) if diff else None
