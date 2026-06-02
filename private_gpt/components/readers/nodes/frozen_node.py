from typing import Any

from pydantic import Field

from private_gpt.components.readers.nodes import TreeNode
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode


class FrozenNode(TreeNode):
    """Allow to have a frozen node version to avoid to have live content.

    This is useful when we want to keep content blocked (after expansion for example),
    to improve performances.
    """

    contents: dict[TreeMetadataMode, str] = Field(
        default_factory=dict,
        description="Contents of the node for each metadata mode.",
    )

    @classmethod
    def from_node(
        cls, node: TreeNode, modes: list[TreeMetadataMode] | None = None
    ) -> "FrozenNode":
        """Create a FrozenNode from a TreeNode."""
        modes = modes or list(TreeMetadataMode)
        return cls(
            id_=node.id_,
            extra_info=node.metadata,
            abs_idx=node.abs_idx,
            idx=node.idx,
            height=node.height,
            depth=node.depth,
            excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
            excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
            contents={mode: node.get_content(mode) for mode in modes},
        )

    def set_content(self, value: Any) -> None:
        """Set the content of the node."""
        # Nothing to do
        pass

    def get_content_internal(
        self, metadata_mode: TreeMetadataMode = TreeMetadataMode.NONE
    ) -> str:
        """Get object content."""
        if len(self.contents) == 1:
            return next(iter(self.contents.values()))

        if metadata_mode not in self.contents:
            return ""

        return self.contents[metadata_mode]
