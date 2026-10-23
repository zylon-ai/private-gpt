from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import BaseNode, TransformComponent

from private_gpt.components.ingest.metadata_helper import MetadataFlags
from private_gpt.components.readers.nodes import SectionNode, TreeNode


class MarkNoPrunableNodesTransform(TransformComponent):
    """As Slides may contain many empty sections, we mark them as non-pruneable."""

    def _update_hidden_elements(self, node: TreeNode) -> None:
        """Update hidden metadata for nodes based on hidden_regex."""
        if isinstance(node, SectionNode):
            if not node.children:
                node.metadata[MetadataFlags.NO_PRUNABLE] = True
                node.excluded_llm_metadata_keys.append(MetadataFlags.NO_PRUNABLE)
                node.excluded_embed_metadata_keys.append(MetadataFlags.NO_PRUNABLE)

        for child in node.children or []:
            self._update_hidden_elements(child)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        for node in nodes:
            if isinstance(node, TreeNode):
                self._update_hidden_elements(node)
        return nodes

    @classmethod
    def from_defaults(
        cls,
    ) -> "MarkNoPrunableNodesTransform":
        return cls()
