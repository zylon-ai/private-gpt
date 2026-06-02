from typing import Any

from private_gpt.components.readers.nodes import DocumentRootNode


class FragmentRootNode(DocumentRootNode):
    """A subclass of DocumentRoot representing the root of a fragment."""

    def model_dump(
        self,
        *,
        include_parent: bool = False,
        include_children: bool = False,
        include_tree: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return super().model_dump(
            include_parent=include_parent,
            include_children=include_children,
            include_tree=include_tree,
            **kwargs,
        )
