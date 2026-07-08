import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

from private_gpt.components.ingest.metadata_helper import MetadataFlags, MetadataNode
from private_gpt.components.readers.nodes import SectionNode, TreeNode
from private_gpt.components.readers.nodes.fragment_node import FragmentRootNode
from private_gpt.settings.settings import settings

debug_mode = settings().server.debug_mode or True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)


class SplitSubtreeAlg:
    """Algorithm to split a tree into subtrees at specified split points.

    The tree is split into subtrees at nodes that are considered split points,
    such as section nodes. The subtrees are created by taking the nodes between
    two split points and creating a new tree structure with the nodes as children
    of a new root node. The new root node is then added to the list of subtrees.
    """

    def split_subtree(self, node: TreeNode) -> list[TreeNode]:
        """Split the tree into subtrees at the specified split points.

        Args:
            node (TreeNode): The node to start splitting from.

        Returns:
            list[TreeNode]: A list of subtrees.
        """
        sorted_nodes = list(node.flatten())
        split_nodes: list[TreeNode] = [
            n for n in sorted_nodes if self._is_split_point(n)
        ]

        # Find split points where sections have siblings
        split_indices: list[int] = []
        for n in split_nodes:
            siblings = n.parent.children if n.parent else []
            if len(siblings) > 1:
                split_indices.append(sorted_nodes.index(n))
        split_indices = sorted(set(split_indices))
        logger.debug(f"Split indices: {split_indices}")

        # Prune first element in a nested object
        # since below content of the section has been joined
        # with some content, and it is the nearest split point
        new_split_indices: list[int] = split_indices.copy()
        for i in split_indices:
            node = sorted_nodes[i]
            siblings = node.parent.children if node.parent else []
            if len(siblings) > 1:
                section_siblings = [n for n in siblings if self._is_split_point(n)]
                if node in section_siblings:
                    idx = section_siblings.index(node)
                    if (
                        idx == 0
                        and node.parent
                        and node.parent.abs_idx in split_indices
                        and node.abs_idx in split_indices
                    ):
                        new_split_indices.remove(node.abs_idx)
                        logger.debug(f"Pruned split index: {node.abs_idx}")
                        continue

            if node.metadata.get(MetadataFlags.NO_PRUNABLE.value):
                new_split_indices.remove(i)
                logger.debug(f"Pruned no-prunable split index: {i}")

        # Split the tree into subtrees at the specified indices
        logger.debug(f"Processed Split indices: {new_split_indices}")
        return self._split_subtrees(sorted_nodes, new_split_indices)

    def _is_split_point(self, node: TreeNode) -> bool:
        """Check if the node is a split point."""
        return node.isinstance(SectionNode)

    def get_root(self, node: TreeNode) -> TreeNode:
        """Get the root node of the tree."""
        while node.parent:
            node = node.parent
        return node

    def _split_subtrees(
        self, nodes: list[TreeNode], split_indices: list[int]
    ) -> list[TreeNode]:
        """Split the tree into subtrees at the specified indices."""
        if not nodes:
            return []
        if not split_indices:
            # We don't have any split points, so return the tree as is
            # processing the result
            split_indices = [0]

        # Split the nodes into subtrees based on the split indices
        subtrees_matrix: list[list[TreeNode]] = []
        start = 0
        current_node: list[TreeNode] = []
        for idx in split_indices:
            subtrees_matrix.append(current_node + nodes[start:idx])
            start = idx + 1
            current_node = [nodes[idx]]
        subtrees_matrix.append(current_node + nodes[start:])

        # Rebuild the subtrees from the split nodes
        subtrees = []
        with ThreadPoolExecutor() as executor:
            results = executor.map(
                lambda subtree_nodes: self._create_subtree(subtree_nodes),
                subtrees_matrix,
            )
            for new_subtree in results:
                if new_subtree:
                    logger.debug(
                        f"Generated a new subtree with {len(new_subtree.children)} nodes"
                    )
                    subtrees.append(new_subtree)
                else:
                    logger.debug("Failed to create subtree. Skipping.")
        return subtrees

    def _create_subtree(
        self,
        nodes: list[TreeNode],
    ) -> TreeNode | None:
        if not nodes:
            return None

        # Step 1: Rebuild the tree structure from the flat list of nodes
        copy_nodes = {node.id_: node.__class__.clone_tree(node) for node in nodes}

        # Step 2: Create a new root node and add the subtrees as children
        root: TreeNode | None = FragmentRootNode(
            id_=str(uuid.uuid4()),
            extra_info={**copy_nodes[nodes[0].id_].metadata},
            excluded_llm_metadata_keys=copy_nodes[
                nodes[0].id_
            ].excluded_llm_metadata_keys,
            excluded_embed_metadata_keys=copy_nodes[
                nodes[0].id_
            ].excluded_embed_metadata_keys,
            abs_idx=max(node.abs_idx for node in nodes),
            idx=min(node.idx for node in nodes),
        )

        # Step 3: Remove any nodes that are not within the subtree
        flatten_list = [list(node.flatten()) for _, node in copy_nodes.items()]
        for node in [item for sublist in flatten_list for item in sublist]:
            indexer = copy_nodes.get(node.id_)
            if not indexer:
                node.parent_id = None
                if node.parent:
                    node.parent.children.remove(node)
                    node.parent = None
                    node.parent_id = None

        # Rebuild tree
        TreeNode.rebuild_tree(list(copy_nodes.values()), root_node=nodes[0])
        if not root:
            return None

        # Concat content with new root
        minimum_depth = min(node.depth for node in copy_nodes.values())
        lower_depth = [
            node for node in copy_nodes.values() if node.depth == minimum_depth
        ]
        for node in lower_depth:
            root.add_child(node)

        # Step 4: Prune the tree to remove any empty nodes
        root = root.prune()
        if not root:
            return None

        # Step final: Update and return data

        # Update token count based on the new tree
        all_token_count = [node.token_count for node in copy_nodes.values()]
        root.metadata[MetadataNode.TOKEN_COUNT.value] = sum(all_token_count)

        return root
