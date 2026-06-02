import bisect
from dataclasses import dataclass
from textwrap import shorten
from typing import Any, Generic, TypeVar

from private_gpt.components.readers.nodes.section_node import SectionNode
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode

_token_metadata_key = "tokens"


@dataclass
class _ExpandingNode:
    node: TreeNode
    distance: int = 0

    def __str__(self) -> str:
        return f"{self.distance}: [{self.node.get_type()}] {shorten(self.node.get_content(TreeMetadataMode.RAG), 50)}"

    def __repr__(self) -> str:
        return self.__str__()


T = TypeVar("T", bound=TreeNode)


class PaperDistanceAlg(Generic[T]):
    debug = False

    huge_jump = 10000000
    large_jump = 100000
    small_jump = 1

    def print(self, *args: Any) -> None:
        if self.debug:
            print(*args)

    def fill_window(self, hit_node: T, remaining_tokens: int) -> set[str]:
        result_node_id_set = set()
        processed_nodes_id_set = set()

        # Reserve the tokens of the hit node and add it to the result unconditionally
        hit_node_tokens = hit_node.token_count or 0
        if hit_node_tokens > remaining_tokens:
            # Edge case where the hit node itself is larger than the window,
            # not much to do here so just return empty set,
            # nodes can't be split further
            return set()
        remaining_tokens -= hit_node.token_count
        result_node_id_set.add(hit_node.id_)
        processed_nodes_id_set.add(hit_node.id_)

        # Reserve the tokens of sections that lead to the hit node
        # These will be added to the result after expanding the window
        # and serve as "tiny summary" of the document itself
        #
        # Title of the document
        # # Section 1
        # ## Subsection 1
        # This text describes ... (hit node)
        # ...
        for path_node in self._path_to_root(hit_node):
            path_item_tokens = path_node.token_count
            if remaining_tokens >= path_item_tokens:
                result_node_id_set.add(path_node.id_)
                remaining_tokens -= path_item_tokens

        nodes_to_explore = [_ExpandingNode(node=hit_node, distance=0)]

        while len(nodes_to_explore) > 0:
            closest_node = nodes_to_explore.pop(0)
            self.print(
                "Closest node:", closest_node, "remaining tokens:", remaining_tokens
            )
            remaining_tokens -= closest_node.node.token_count
            if remaining_tokens >= 0:
                result_node_id_set.add(closest_node.node.id_)
            else:
                self.print(
                    "Window full, remaining tokens:",
                    remaining_tokens + closest_node.node.token_count,
                    "closest node tokens:",
                    closest_node.node.token_count,
                    "closest node:",
                    closest_node,
                )
                break  # Window is full, the closest node doesn't fit anymore

            connected_nodes = self._find_weighted_connected_nodes(closest_node)
            for connected_node in connected_nodes:
                if connected_node.node.id_ in processed_nodes_id_set:
                    continue  # Skip already visited nodes to avoid the insort cost ahead of time
                processed_nodes_id_set.add(connected_node.node.id_)
                # self.print("--> Connected node:", connected_node)
                bisect.insort(
                    nodes_to_explore,
                    connected_node,
                    key=lambda x: x.distance,
                )
        return result_node_id_set

    @staticmethod
    def _path_to_root(node: TreeNode | None) -> list[TreeNode]:
        path = []
        while True and node:
            node = node.parent
            if node:
                path.append(node)
            else:
                break
        return path

    def _find_weighted_connected_nodes(
        self, node: _ExpandingNode
    ) -> list[_ExpandingNode]:
        """Find connected nodes with weighted distances.

        This is essentially a graph traversal with weighted edges.

               ------ [parent] -----
              |          |         |
          [sibling] - [node] - [sibling]
                        |
                     [child] ... [child-1] ... [child-n]

        All directions are explored and given a distance, later
        these nodes are kept in a sorted list so that the closest
        ones to keep expanding the window are always at the front.

        The distance to move to siblings, parent and children is weighted.

        In practice this means that sections of a document from the hit node
        will be expanded before attempting to move to another section.
        """
        connected_nodes = []
        if node.node.parent:
            parent_with_distance = _ExpandingNode(
                node=node.node.parent,
                distance=node.distance
                + self._weighted_jump(node.node, node.node.parent, "up"),
            )
            connected_nodes.append(parent_with_distance)

            sibling_nodes = node.node.parent.children
            # The position within the children list
            # [0] [1] [left]-[node]-[right] [5] [6]
            index_of_node_among_siblings = sibling_nodes.index(node.node)

            if index_of_node_among_siblings < len(sibling_nodes) - 1:
                right_sibling = sibling_nodes[index_of_node_among_siblings + 1]
                right_sibling_with_distance = _ExpandingNode(
                    node=right_sibling,
                    distance=node.distance
                    + self._weighted_jump(node.node, right_sibling, "right"),
                )
                connected_nodes.append(right_sibling_with_distance)

            if index_of_node_among_siblings > 0:
                left_sibling = sibling_nodes[index_of_node_among_siblings - 1]
                left_sibling_with_distance = _ExpandingNode(
                    node=left_sibling,
                    distance=node.distance
                    + self._weighted_jump(node.node, left_sibling, "left"),
                )
                connected_nodes.append(left_sibling_with_distance)

        if node.node.children and len(node.node.children) > 0:
            for _, child in enumerate(node.node.children):
                child_with_distance = _ExpandingNode(
                    node=child,
                    distance=node.distance
                    + self._weighted_jump(node.node, child, "down"),
                )
                connected_nodes.append(child_with_distance)

        return connected_nodes

    def _weighted_jump(
        self, from_node: TreeNode, to_node: TreeNode, direction: str
    ) -> int:
        """Calculate the weighted distance between two nodes.

        These jump is based on the type of the node so that some rules are followed:
        - Going from SectionNode to SectionNode
            is very expensive
        - Going from SectionNode to any non-SectionNode
            is the least expensive (going to the content)
        - Going from any non-SectionNode to any other non-SectionNode
            is cheap (moving within the content)
        - Going from any non-SectionNode to SectionNode
            is expensive (moving up to a new section)
        """
        from_section = from_node.isinstance(SectionNode)
        from_content = not from_section
        to_section = to_node.isinstance(SectionNode)
        to_content = not to_section

        distance = 1
        if direction == "left" or direction == "right":
            if from_section and to_section:
                distance = self.large_jump  # Moving horizontally between sections
            elif from_section and to_content:
                # This case should be rare, plain content at the root
                # followed by a section since we are in the horizontal scenario
                distance = self.small_jump
            elif from_content and to_content:
                distance = self.small_jump  # Horizontal movement within the content
            elif from_content and to_section:
                distance = self.small_jump  # Moving to a subsection within the content
        elif direction == "up":
            distance = self.huge_jump  # Going up in the hierarchy is the most expensive
        elif direction == "down":
            distance = self.small_jump

        return distance
