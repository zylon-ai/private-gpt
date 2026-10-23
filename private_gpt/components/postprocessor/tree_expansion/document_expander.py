import logging
from enum import Enum, auto
from typing import Any, Protocol

from private_gpt.components.readers.nodes.section_node import SectionNode
from private_gpt.components.readers.nodes.tree_node import TreeNode
from private_gpt.settings.settings import settings

debug_mode = settings().server.debug_mode

logger = logging.getLogger(__name__)


class ExpansionResult(Enum):
    CONTINUE = auto()  # Can continue expanding other parts
    STOP = auto()  # Stop further expansion due to token limit
    ROLLBACK = auto()  # Skip this node, but can continue with others

    def is_success(self) -> bool:
        return self in [ExpansionResult.CONTINUE]

    def is_failure(self) -> bool:
        return self in [ExpansionResult.STOP, ExpansionResult.ROLLBACK]


class ExpansionDirection(Enum):
    DOWNWARDS = auto()
    HORIZONTALLY = auto()
    UPWARDS = auto()


class ExpansionDirectionProtocol(Protocol):
    def __call__(self, current_node: TreeNode, **kwargs: Any) -> ExpansionResult: ...


class DocumentTreeExpander:
    """Expands a document tree while respecting token limits.

    This class implements a strategy to expand a document tree in multiple directions
    (downwards, horizontally, and upwards) while ensuring the total token count stays
    within specified limits. It uses different failure handling strategies for different
    expansion phases.
    """

    debug: bool = debug_mode

    node: TreeNode
    max_tokens: int

    processed_nodes_ids: dict[ExpansionDirection, set[str]]
    result_nodes_ids: set[str]

    _current_success_result: ExpansionResult = ExpansionResult.CONTINUE
    _current_failure_result: ExpansionResult = ExpansionResult.STOP

    _remaining_tokens: int
    _token_counts: dict[str, int]

    huge_jump = 10000000
    large_jump = 100000
    small_jump = 1

    def __init__(
        self,
        node: TreeNode,
        max_tokens: int,
        processed_nodes_ids: dict[ExpansionDirection, set[str]] | None = None,
    ) -> None:
        """Initialize the document tree expander.

        Args:
            node: The node of the tree to expand
            max_tokens: Maximum number of tokens to expand
            processed_nodes_ids: Dictionary mapping direction to sets of IDs
        """
        self.node = node
        self.max_tokens = max_tokens
        self._remaining_tokens = max_tokens
        self.processed_nodes_ids = processed_nodes_ids or {
            ExpansionDirection.DOWNWARDS: set(),
            ExpansionDirection.HORIZONTALLY: set(),
            ExpansionDirection.UPWARDS: set(),
        }
        self.result_nodes_ids = set()
        self._token_counts = {}

    def fill_window(self) -> tuple[set[str], int]:
        """Main method to expand the document tree within token constraints.

        The expansion happens in two phases:
        1. Downward expansion with STOP
            on failure to preserve partial subtrees.
        2. Horizontal and upward expansion with ROLLBACK
            on failure for complete subtrees.

        Returns:
            Set of expanded node IDs
        """
        # Start expanding recursively in all directions
        result = self._expand_recursively(
            current_node=self.node,
            include_downwards=True,
            include_horizontally=True,
            include_upwards=True,
        )
        token_count = self.max_tokens - self._remaining_tokens
        if result.is_failure():
            return self.result_nodes_ids, token_count

        # Return the IDs of the expanded nodes
        logger.debug(f"Expanded nodes: {self.result_nodes_ids}")
        return self.result_nodes_ids, token_count

    def _get_token_count_by_id(self, node_id: str) -> int | None:
        """Calculate the token count of a node and its subtree (if applicable).

        Args:
            node_id: The node ID to count tokens for
            include_subtree: Whether to include tokens from all descendant nodes

        Returns:
            Total token count
        """
        return self._token_counts.get(node_id)

    def _get_token_count(self, node: TreeNode, include_subtree: bool) -> int:
        """Calculate the token count of a node and its subtree (if applicable).

        Args:
            node: The node to count tokens for
            include_subtree: Whether to include tokens from all descendant nodes

        Returns:
            Total token count
        """

        def get_key() -> str:
            suffix = "-subtree" if include_subtree else None
            return f"{node.id_}{suffix}"

        key = get_key()
        if key in self._token_counts:
            return self._token_counts[key]
        count = node.token_count if not include_subtree else node.get_sum_token_count()
        self._token_counts[key] = count
        return count

    def _is_already_processed(
        self, node: TreeNode, direction: ExpansionDirection | None
    ) -> bool:
        """Check if a node has already been processed in a specific direction.

        Args:
            node: Node to check
            direction: Direction to check for

        Returns:
            True if the node has been processed, False otherwise
        """
        if direction is not None:
            return node.id_ in self.processed_nodes_ids[direction]

        all_full_processed = self.processed_nodes_ids[
            ExpansionDirection.DOWNWARDS
        ].intersection(
            self.processed_nodes_ids[ExpansionDirection.HORIZONTALLY],
            self.processed_nodes_ids[ExpansionDirection.UPWARDS],
        )
        return node.id_ in all_full_processed

    def _expand_direction(
        self,
        current_node: TreeNode,
        expand_func: ExpansionDirectionProtocol,
        direction: ExpansionDirection,
        **kwargs: Any,
    ) -> ExpansionResult:
        """Expand a node in a specific direction and process the result.

        Args:
            current_node: Node to expand
            expand_func: Function implementing the expansion strategy
            direction: Direction of expansion
            kwargs: Additional arguments for the expansion function

        Returns:
            Result of the expansion attempt
        """
        reverse = kwargs.pop("reverse", False)

        # Only add to results immediately if moving right
        if not reverse:
            self.result_nodes_ids.add(current_node.id_)

        # Step 2: Expand the node in the specified direction
        result = expand_func(current_node, reverse=reverse, **kwargs)

        # Step 1: Mark the node as processed in the specified direction
        self.processed_nodes_ids[direction].add(current_node.id_)

        # Only add to results if all subtrees are processed
        if reverse:
            self.result_nodes_ids.add(current_node.id_)

        # Step 3: Process the result
        if result == ExpansionResult.STOP:
            logger.debug(f"Stopped at node: {current_node}")
            # If we're moving left and hit a stop, we should remove this node
            # if not all children are in the result set
            all_children = [node.id_ for node in current_node.children]
            if reverse and any(
                node_id not in self.result_nodes_ids for node_id in all_children
            ):
                self.result_nodes_ids.discard(current_node.id_)

        elif result == ExpansionResult.CONTINUE:
            logger.debug(f"Expanded node: {current_node}")

        return result

    def _expand_recursively(
        self,
        current_node: TreeNode,
        include_downwards: bool = False,
        include_horizontally: bool = False,
        include_upwards: bool = False,
        **kwargs: Any,
    ) -> ExpansionResult:
        """Recursive method to expand nodes while respecting token constraints.

        Args:
            current_node: Node to potentially expand
            include_downwards: Flag to expand children
            include_horizontally: Flag to expand siblings
            include_upwards: Flag to expand parents
            kwargs: Additional arguments to pass to the expansion functions

        Returns:
            Result indicating if expansion was successful
        """
        # Check if node has been processed in all directions
        if self._is_already_processed(current_node, direction=None):
            logger.debug(f"Skipping already processed node: {current_node}")
            return self._current_success_result

        # Check token limit
        potential_tokens = self._get_token_count(current_node, include_subtree=False)
        if potential_tokens > self._remaining_tokens:
            logger.debug(f"Exceeded token limit at node: {current_node}")
            return self._current_failure_result

        # Update remaining tokens before expansion
        self._remaining_tokens -= potential_tokens

        # First, try to expand downwards (depth-first)
        if include_downwards:
            logger.debug(f"Expanding node in direction downwards: {current_node}")
            expansion_success = self._expand_direction(
                current_node=current_node,
                expand_func=self._expand_downwards,
                direction=ExpansionDirection.DOWNWARDS,
                **kwargs,
            )
            if expansion_success.is_failure():
                return expansion_success

        if include_horizontally:
            logger.debug(f"Expanding node in direction horizontally: {current_node}")
            expansion_success = self._expand_direction(
                current_node=current_node,
                expand_func=self._expand_horizontally,
                direction=ExpansionDirection.HORIZONTALLY,
                **kwargs,
            )
            if expansion_success.is_failure():
                return expansion_success

        # If horizontal expansion is complete, try expanding upwards
        if include_upwards:
            logger.debug(f"Expanding node in direction upwards: {current_node}")
            expansion_success = self._expand_direction(
                current_node=current_node,
                expand_func=self._expand_upwards,
                direction=ExpansionDirection.UPWARDS,
                **kwargs,
            )
            if expansion_success.is_failure():
                return expansion_success

        return self._current_success_result

    def _expand_downwards(
        self, current_node: TreeNode, **kwargs: Any
    ) -> ExpansionResult:
        """Expand node's children depth-first.

        Args:
            current_node: Node whose children to expand
            kwargs: Additional expansion parameters

        Returns:
            Result of the downward expansion
        """
        if self._is_already_processed(current_node, ExpansionDirection.DOWNWARDS):
            logger.debug(f"Skipping downwards expansion for node: {current_node}")
            return self._current_success_result

        reverse = kwargs.get("reverse", False)
        children = reversed(current_node.children) if reverse else current_node.children

        for child in children:
            if self._is_already_processed(child, ExpansionDirection.DOWNWARDS):
                continue

            # Check if adding this child would exceed token limit
            potential_tokens = self._get_token_count(
                child,
                include_subtree=True,
            )
            if potential_tokens > self._remaining_tokens:
                return self._current_failure_result

            # Attempt to expand child
            child_expansion = self._expand_recursively(
                current_node=child,
                include_downwards=True,
                include_horizontally=True,
                include_upwards=False,
                **kwargs,
            )
            if child_expansion.is_failure():
                return child_expansion

        return self._current_success_result

    def _expand_horizontally(
        self, current_node: TreeNode, **kwargs: dict[str, Any]
    ) -> ExpansionResult:
        """Expand sibling nodes with weighted ping-pong strategy."""
        if not current_node.parent:
            logger.debug(f"Skipping horizontal expansion for root node: {current_node}")
            return self._current_success_result
        if self._is_already_processed(
            current_node.parent, ExpansionDirection.HORIZONTALLY
        ):
            logger.debug(
                f"Skipping horizontal expansion for node: {current_node.parent}"
            )
            return self._current_success_result

        # Calculate sibling weights based on node types and content
        siblings = current_node.parent.children
        current_index = siblings.index(current_node)

        # Ping-pong expansion strategy
        right_candidates = siblings[current_index + 1 :]
        left_candidates = siblings[:current_index][::-1]

        # Normalize the lengths of right and left candidates
        max_length = max(len(right_candidates), len(left_candidates))
        normalized_right = right_candidates + [None] * (
            max_length - len(right_candidates)
        )
        normalized_left = left_candidates + [None] * (max_length - len(left_candidates))

        # Last successfully expanded nodes in each direction
        right_expanded, left_expanded = current_node, current_node

        for right, left in zip(normalized_right, normalized_left, strict=False):
            # Calculate candidate list
            candidates: list[TreeNode] = [
                n
                for n in [right, left]
                if n
                and not self._is_already_processed(n, ExpansionDirection.HORIZONTALLY)
                and n not in (right_expanded, left_expanded)
            ]
            if not candidates:
                continue

            # Sort candidates considering:
            # 1. Weight (lower is better)
            # 2. Token count (lower is better)
            # 3. Distance from current node (lower is better)
            # 4. Move before to righter than left
            candidates.sort(
                key=lambda sibling: (
                    self._calculate_node_weight(current_node, sibling),
                    self._get_token_count(sibling, include_subtree=True),
                    abs(siblings.index(sibling) - current_index),
                    siblings.index(sibling) > current_index,
                )
            )

            results: list[ExpansionResult] = []
            for sibling in candidates:
                # Only expand downwards to avoid infinite loops
                expanded_siblings = self._expand_recursively(
                    sibling,
                    include_downwards=True,
                    include_horizontally=False,
                    include_upwards=False,
                    reverse=kwargs.pop("reverse", False) or sibling == left,
                    **kwargs,
                )
                results.append(expanded_siblings)
                if expanded_siblings.is_failure():
                    continue

                # Update references for ping-pong expansion
                right_expanded, left_expanded = (
                    sibling if sibling == right else right_expanded,
                    sibling if sibling == left else left_expanded,
                )

            # If all candidates failed, we stop the expansion
            if all(result.is_failure() for result in results):
                return self._current_failure_result

        # If all candidates succeeded, we continue the expansion
        return self._current_success_result

    def _expand_upwards(self, current_node: TreeNode, **kwargs: Any) -> ExpansionResult:
        """Attempt to expand upwards through parent nodes.

        Args:
            current_node: Node whose parent to expand
            kwargs: Additional expansion parameters

        Returns:
            Result of the upward expansion
        """
        parent = current_node.parent
        if not parent:
            # Stop if we reach the root node
            logger.debug(f"Reached root node: {current_node}")
            return self._current_failure_result
        if self._is_already_processed(parent, ExpansionDirection.UPWARDS):
            logger.debug(f"Skipping upwards expansion for node: {parent}")
            return self._current_success_result

        # Expand parent node
        result = self._expand_recursively(
            current_node=parent,
            include_downwards=True,
            include_horizontally=True,
            include_upwards=True,
            **kwargs,
        )

        return result

    def _calculate_node_weight(self, from_node: TreeNode, to_node: TreeNode) -> float:
        """Calculate weight for node expansion based on type and subtree.

        Args:
            from_node: Node from which to expand
            to_node: Node to which to expand

        Returns:
            Weight multiplier for node expansion
        """
        from_section = from_node.isinstance(SectionNode)
        from_content = not from_section
        to_section = to_node.isinstance(SectionNode)
        to_content = not to_section

        distance = self.small_jump
        if from_section and to_section:
            # Moving horizontally between sections
            distance = self.large_jump
        elif from_section and to_content:
            # This case should be rare, plain content at the root
            # followed by a section since we are in the horizontal scenario
            distance = self.small_jump
        elif from_content and to_content:
            # Horizontal movement within the content
            distance = self.small_jump
        elif from_content and to_section:
            # Moving to a subsection within the content
            distance = self.small_jump

        return distance
