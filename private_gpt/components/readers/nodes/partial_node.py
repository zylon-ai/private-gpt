import builtins
from typing import Any, Self

from pydantic import Field

from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode


class PartialNode(TreeNode):
    type: str = Field(description="Node type.")
    original_hash: str | None = Field(
        default=None,
        description="Hash of the original node, if available.",
        alias="hash",
    )

    def get_content_internal(
        self, metadata_mode: TreeMetadataMode = TreeMetadataMode.ALL
    ) -> str:
        if (
            metadata_mode != TreeMetadataMode.NONE
            and metadata_mode != TreeMetadataMode.RAG
        ):
            if self.children:
                return "".join(
                    child.get_content(metadata_mode) for child in self.children
                )
        return ""

    def set_content(self, value: Any) -> None:
        pass

    @property
    def hash(self) -> str:
        return self.original_hash or super().hash

    def __str__(self) -> str:
        content = self.token_count
        return (
            f"{self.type}({self.id_})" + f": Token count {content}" if content else ""
        )

    def __repr__(self) -> str:
        return self.__str__()

    def model_dump(
        self,
        *,
        include_parent: bool = False,
        include_children: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return super().model_dump(
            include_parent=include_parent,
            include_children=include_children,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, data: builtins.dict[str, Any], **kwargs: Any) -> Self:
        return super(TreeNode, cls).from_dict(data, **kwargs)

    @classmethod
    def from_node(cls, node: TreeNode) -> "PartialNode":
        if isinstance(node, PartialNode):
            # Already a partial node
            return node

        partial_node = cls(
            type=node.get_type(),
            hash=node.hash,
            **node.dict(include_parent=False, include_children=False),
        )

        for child in node.children:
            partial_node.add_child(cls.from_node(child))

        return partial_node

    @classmethod
    def from_partial_dict(cls, node_dict: builtins.dict[str, Any]) -> "PartialNode":
        children_dict = node_dict.pop("children", [])
        children: list[PartialNode] = [
            cls.from_partial_dict(child_dict) for child_dict in children_dict
        ]
        node = cls.from_dict(node_dict)
        for child in children:
            node.children.append(child)
            child.parent = node
        return node

    def isinstance(
        self, cls: builtins.type[TreeNode] | tuple[builtins.type[TreeNode], ...]
    ) -> bool:
        if isinstance(cls, tuple):
            return any(self.isinstance(c) for c in cls)
        tree_type = cls.get_type() if cls and issubclass(cls, TreeNode) else None
        return tree_type == self.type
