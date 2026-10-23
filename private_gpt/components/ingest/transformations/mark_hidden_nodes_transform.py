import re
from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import BaseNode, TransformComponent
from pydantic import Field

from private_gpt.components.ingest.metadata_helper import MetadataFlags
from private_gpt.components.readers.nodes import TextNode, TreeNode


class MarkHiddenNodesTransform(TransformComponent):
    hidden_regex: re.Pattern[str] = Field(
        default=re.compile(r"^\s*<!--.*?-->\s*$", re.MULTILINE),
        description="Regex pattern to identify hidden HTML comments.",
        exclude=True,
    )

    def _update_hidden_elements(self, node: TreeNode) -> None:
        """Update hidden metadata for nodes based on hidden_regex."""
        if isinstance(node, TextNode):
            if self.hidden_regex.match(node.text):
                node.metadata[MetadataFlags.HIDDEN.value] = True
                node.excluded_llm_metadata_keys.append(MetadataFlags.HIDDEN.value)
                node.excluded_embed_metadata_keys.append(MetadataFlags.HIDDEN.value)

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
        hidden_regex: str = r"^\s*<!--.*?-->\s*$",
    ) -> "MarkHiddenNodesTransform":
        return cls(
            hidden_regex=re.compile(hidden_regex, re.MULTILINE),
        )
