import logging
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, cast

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import (
    BaseNode,
    MetadataMode,
    NodeWithScore,
    QueryBundle,
)
from pydantic import BaseModel, PrivateAttr

from private_gpt.components.ingest.metadata_helper import MetadataChunk, MetadataNode
from private_gpt.components.llm.llm_helper import TokenizerFn
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.postprocessor.tree_expansion.document_expander import (
    DocumentTreeExpander,
)
from private_gpt.components.postprocessor.tree_expansion.split_subtrees import (
    SplitSubtreeAlg,
)
from private_gpt.components.readers.nodes.document_node import DocumentRootNode
from private_gpt.components.readers.nodes.frozen_node import FrozenNode
from private_gpt.components.readers.nodes.partial_node import PartialNode
from private_gpt.components.readers.nodes.tree_node import TreeNode
from private_gpt.settings.settings import settings
from private_gpt.utils.random import generate_deterministic_uuid_from_seed

if TYPE_CHECKING:
    from concurrent.futures import Future

config = settings()
debug_mode = config.server.debug_mode

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)


class ExpansionResult(BaseModel):
    """Result of expanding a node into a subtree."""

    node_ids: set[str]
    token_count: int


class NodeSelection(BaseModel):
    """Represents a selection of nodes chosen for processing."""

    hit_nodes: list[NodeWithScore]
    root_nodes: list[DocumentRootNode]
    subtrees: list[ExpansionResult]

    @property
    def used_token_limit(self) -> int:
        return sum([subtree.token_count for subtree in self.subtrees])


class ProcessedNodes(BaseModel):
    """Contains data about processed nodes."""

    filtered_items: list[tuple[NodeWithScore, DocumentRootNode, set[str]]]
    loaded_nodes: dict[str, TreeNode]


class EvaluationResult(BaseModel):
    """Result of evaluating a specific configuration."""

    score: float
    node_selection: NodeSelection


class TreeExpansionReplacementPostProcessor(BaseNodePostprocessor, ABC):
    # Retriever config
    node_component: NodeStoreComponent
    collection: str

    # Token limit config
    token_limit: int | None = None
    tokenizer_fn: TokenizerFn | None = None

    # Maximize top_k
    maximize_top_k: bool = config.retrieval.maximize_top_k
    min_top_k: int = 2**2
    max_top_k: int = config.retrieval.top_k
    step: int = 2**2
    max_patience: int = 2

    # Weighting config
    relevance_weight: float = 0.2
    token_weight: float = 0.5
    entropy_weight: float = 0.3

    # Merging config
    overlap_threshold: float = 0
    max_recalculations: int = config.retrieval.max_merging_recalculations

    _root_nodes: dict[str, DocumentRootNode] = PrivateAttr(default_factory=dict)

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        """Create a expansion pipeline trying to maximize the content and relevance.

        Steps:
        1. Find optimal set of nodes using a greedy algorithm
        2. Process the optimal set of nodes
        3. Generate final result nodes
        """
        if not nodes:
            return []

        # Initialize setup
        absolute_token_limit = self.token_limit or 0
        logger.debug(f"Processing nodes with token limit: {absolute_token_limit}")

        # =====================================================================
        # STEP 1: Find optimal set of nodes through search
        # =====================================================================
        node_selection = self._find_optimal_nodes(nodes, absolute_token_limit)
        logger.debug(
            f"Optimal set found: {len(node_selection.hit_nodes)} nodes selected"
        )

        # =====================================================================
        # STEP 2: Process the expanded nodes from optimal set
        # =====================================================================
        processed_nodes = self._process_expanded_nodes(node_selection)
        logger.debug(f"Processed {len(processed_nodes.filtered_items)} expanded nodes")

        # =====================================================================
        # STEP 3: Generate final result nodes
        # =====================================================================
        result_nodes = self._generate_result_nodes(processed_nodes)
        logger.debug(f"Generated {len(result_nodes)} final result nodes")

        return result_nodes

    # ===========================================================================
    # STEP 1 METHODS: Finding optimal set of nodes
    # ===========================================================================
    def _find_optimal_nodes(
        self, nodes: list[NodeWithScore], absolute_token_limit: int
    ) -> NodeSelection:
        """Find the optimal set of nodes through greedy search.

        Returns:
            NodeSelection: The optimal selection of nodes
        """
        # Filter and sort nodes
        all_hit_nodes = self._prepare_hit_nodes(nodes)

        # Load root nodes once for efficiency
        self._root_nodes = self._root_nodes or self._load_root_nodes(all_hit_nodes)

        # Run greedy search to find optimal nodes
        return self._greedy_search(all_hit_nodes, absolute_token_limit)

    def _prepare_hit_nodes(self, nodes: list[NodeWithScore]) -> list[NodeWithScore]:
        """Prepare and sort hit nodes."""
        hit_nodes = [node for node in nodes if isinstance(node.node, TreeNode)]
        return sorted(hit_nodes, key=lambda x: float(x.score or 0), reverse=True)

    def _load_root_nodes(
        self, hit_nodes: list[NodeWithScore]
    ) -> dict[str, DocumentRootNode]:
        """Load root nodes for the hit nodes."""
        root_ids = [
            node.node.root_id for node in hit_nodes if hasattr(node.node, "root_id")
        ]
        unique_root_ids = {root_id for root_id in root_ids if root_id}

        root_nodes_map = {
            node.node_id: cast(DocumentRootNode, node)
            for node in self.node_component.get_sorted_nodes(
                collection=self.collection,
                node_ids=list(unique_root_ids),
                limit=len(unique_root_ids),
            )
        }
        logger.debug(f"Loaded {len(root_nodes_map)} root nodes")
        return root_nodes_map

    def _greedy_search(
        self, all_hit_nodes: list[NodeWithScore], absolute_token_limit: int
    ) -> NodeSelection:
        """Perform greedy search to find optimal nodes.

        This considers both top_k and the effect of merging when evaluating
        different configurations.
        """
        # Configuration
        if self.maximize_top_k:
            # Create a linear range from min to max
            min_top_k = min(self.min_top_k, len(all_hit_nodes))
            max_top_k = min(self.max_top_k, len(all_hit_nodes))
            step = self.step
        else:
            # Use a fixed top_k value (len(nodes)=min_top_k=max_top_k)
            min_top_k = len(all_hit_nodes)
            max_top_k = min_top_k
            step = 1

        logger.debug(f"Starting greedy search (range: {min_top_k}-{max_top_k})")

        best_score = 0.0
        best_selection = NodeSelection(hit_nodes=[], root_nodes=[], subtrees=[])

        if len(all_hit_nodes) == 0:
            return best_selection

        current_k = min_top_k
        patience = 0
        while current_k <= max_top_k:
            evaluation_result = self._evaluate_configuration(
                all_hit_nodes, current_k, absolute_token_limit
            )
            logger.debug(f"top_k={current_k}: score={evaluation_result.score:.2f}")

            if evaluation_result.score > best_score:
                best_score = evaluation_result.score
                best_selection = evaluation_result.node_selection
                patience = 0
            elif evaluation_result.score == best_score:
                logger.debug("Same score, reached optimal configuration")
                break
            else:
                patience += 1

            current_k += step
            if current_k > max_top_k:
                logger.debug("Reached max top_k, stopping search")
                break

            if patience >= self.max_patience:
                logger.debug(
                    f"No significant improvement for {patience} iterations, stopping search"
                )
                break

        logger.debug(f"Greedy search complete, best score: {best_score:.2f}")
        return best_selection

    def _evaluate_configuration(
        self,
        all_hit_nodes: list[NodeWithScore],
        top_k: int,
        absolute_token_limit: int,
        current_merge_iteration: int = 0,
    ) -> EvaluationResult:
        """Evaluate a configuration with a specific top_k value.

        Returns:
            EvaluationResult: Result of the evaluation
        """
        # Select nodes and calculate token limits
        hit_nodes = all_hit_nodes[:top_k]
        token_limits = self._calculate_token_limits(hit_nodes, absolute_token_limit)

        # Get root nodes and expand
        root_nodes = self._get_root_nodes_for_hits(hit_nodes)
        expansion_results = self._expand_nodes(hit_nodes, root_nodes, token_limits)

        # Try merging subtrees
        merged_expansion_results = self._merge_subtrees(expansion_results)
        node_selection = NodeSelection(
            hit_nodes=hit_nodes,
            root_nodes=root_nodes,
            subtrees=merged_expansion_results,
        )

        improved_by_merging = any(
            len(result.node_ids) == 0 for result in merged_expansion_results
        )

        # Recalculate if merging is removing nodes
        # to try to improve the score
        if improved_by_merging and current_merge_iteration < self.max_recalculations:
            logger.debug(
                f"Recalculating due to merging (iteration {current_merge_iteration + 1}/{self.max_recalculations})"
            )

            merged_hit_nodes = self._prioritize_merged_nodes(hit_nodes, node_selection)
            if merged_hit_nodes and len(merged_hit_nodes) != len(hit_nodes):
                return self._evaluate_configuration(
                    merged_hit_nodes,
                    len(merged_hit_nodes),
                    absolute_token_limit,
                    current_merge_iteration + 1,
                )

        # Calculate score
        score = self._calculate_configuration_score(
            node_selection, absolute_token_limit
        )

        return EvaluationResult(
            score=score,
            node_selection=node_selection,
        )

    def _calculate_token_limits(
        self, hit_nodes: list[NodeWithScore], absolute_token_limit: int
    ) -> list[int]:
        """Calculate token limits for each node based on its score."""
        score_total = sum(node.score or 0 for node in hit_nodes)
        if score_total == 0:
            # Avoid division by zero
            return [absolute_token_limit // len(hit_nodes) for _ in hit_nodes]

        return [
            int(absolute_token_limit * ((node.score or 0) / score_total))
            for node in hit_nodes
        ]

    def _get_or_load_root_nodes(
        self, node_ids: list[str]
    ) -> dict[str, DocumentRootNode]:
        """Get root nodes from cache or load missing ones from node store.

        Args:
            node_ids: List of root node IDs to retrieve

        Returns:
            dict[str, DocumentRootNode]: Dictionary mapping node IDs to root nodes
        """
        missing_ids = [
            node_id for node_id in node_ids if node_id not in self._root_nodes
        ]
        if missing_ids:
            logger.debug(f"Loading {len(missing_ids)} missing root nodes")
            loaded_nodes = {
                node.node_id: cast(DocumentRootNode, node)
                for node in self.node_component.get_sorted_nodes(
                    collection=self.collection,
                    node_ids=missing_ids,
                    limit=len(missing_ids),
                )
            }
            self._root_nodes.update(loaded_nodes)
        return {
            node_id: self._root_nodes[node_id]
            for node_id in node_ids
            if node_id in self._root_nodes
        }

    def _get_root_nodes_for_hits(
        self, hit_nodes: list[NodeWithScore]
    ) -> list[DocumentRootNode]:
        """Get root nodes for the given hit nodes."""
        root_ids = [
            node.node.root_id
            for node in hit_nodes
            if hasattr(node.node, "root_id") and node.node.root_id
        ]
        root_nodes_map = self._get_or_load_root_nodes(root_ids)
        result = [
            root_nodes_map.get(node.node.root_id)
            for node in hit_nodes
            if hasattr(node.node, "root_id") and node.node.root_id in root_nodes_map
        ]
        return [node for node in result if node]

    def _expand_nodes(
        self,
        hit_nodes: list[NodeWithScore],
        root_nodes: list[DocumentRootNode],
        token_limits: list[int],
    ) -> list[ExpansionResult]:
        """Expand nodes to get subtrees with token counts.

        Returns:
            list[ExpansionResult]: List of expansion results
        """
        with ThreadPoolExecutor() as executor:
            work_items = list(zip(hit_nodes, root_nodes, token_limits, strict=False))

            def process_node(
                work_item: tuple[NodeWithScore, TreeNode, int]
            ) -> ExpansionResult:
                hit_node, root_node, token_limit = work_item
                partial_hit_node = root_node.find_self_or_child_by_id(hit_node.node.id_)

                if partial_hit_node:
                    expansion_result = self._expand(
                        hit_node=partial_hit_node, token_limit=token_limit
                    )
                    return expansion_result

                return ExpansionResult(node_ids=set(), token_count=0)

            return list(executor.map(process_node, work_items))

    def _calculate_configuration_score(
        self, node_selection: NodeSelection, absolute_token_limit: int
    ) -> float:
        """Calculate score for a configuration with dispersion penalty."""

        def collect_node_data() -> tuple[int, float, list[float]]:
            """Collect token count, relevance, and scores from valid nodes."""
            tokens = 0
            relevance = 0.0
            scores = []

            for i, subtree in enumerate(node_selection.subtrees):
                if i < len(node_selection.hit_nodes) and subtree.node_ids:
                    tokens += subtree.token_count
                    score = node_selection.hit_nodes[i].score or 0
                    relevance += subtree.token_count * score
                    scores.append(score)

            return tokens, relevance, scores

        def check_token_limit(tokens: int) -> bool:
            """Verify we don't exceed token limit."""
            return tokens / absolute_token_limit <= 1.0

        def calculate_dispersion(scores: list[float]) -> float:
            """Calculate dispersion penalty based on score variation."""
            if len(scores) <= 1:
                return 0.0

            import numpy as np

            mean = np.mean(scores)

            if mean <= 0:
                return 0.0

            std_dev = np.std(scores)
            cv = std_dev / mean

            return float(cv)

        def calculate_utilization(tokens: int) -> float:
            """Calculate token utilization score."""
            if absolute_token_limit == 0:
                return 0.0
            utilization = tokens / absolute_token_limit
            return 1.0 - abs(0.9 - utilization)

        total_tokens, total_relevance, valid_scores = collect_node_data()
        if not check_token_limit(total_tokens):
            return 0.0

        dispersion_penalty = calculate_dispersion(valid_scores)
        utilization_score = calculate_utilization(total_tokens)
        final_score = (
            +utilization_score * self.token_weight
            + total_relevance * self.relevance_weight
            - dispersion_penalty * self.entropy_weight
        )

        if final_score < 0:
            return 0.0

        logger.debug(
            f"Score: {final_score:.3f} (utilization={utilization_score:.3f}, "
            f"relevance={total_relevance:.3f}, dispersion={dispersion_penalty:.3f})"
        )

        return final_score

    def _merge_subtrees(
        self, expansion_results: list[ExpansionResult]
    ) -> list[ExpansionResult]:
        """Merge overlapping subtrees.

        Args:
            expansion_results: List of expansion results

        Returns:
            list[ExpansionResult]: Updated expansion results after merging
        """
        # Create copies to avoid modifying originals
        results = [
            ExpansionResult(node_ids=set(exp.node_ids), token_count=exp.token_count)
            for exp in expansion_results
        ]

        for i, current in enumerate(results):
            for j, other in enumerate(results):
                if i == j or not current.node_ids or not other.node_ids:
                    continue

                if current.node_ids == other.node_ids:
                    # Same element or duplicate
                    other.node_ids = set()  # Clear
                    other.token_count = 0
                    continue

                intersection = current.node_ids.intersection(other.node_ids)
                if (
                    len(current.node_ids) == 0
                ):  # Skip if current is empty (already merged)
                    continue

                intersection_percentage = len(intersection) / len(current.node_ids)

                if intersection_percentage > self.overlap_threshold:
                    current.node_ids.update(other.node_ids)
                    current.token_count = (  # Estimated
                        current.token_count + other.token_count
                    )

                    other.node_ids = set()
                    other.token_count = 0
                elif intersection_percentage > 0:
                    other.node_ids.difference_update(intersection)
                    other.token_count = int(
                        other.token_count
                        * (
                            len(other.node_ids)
                            / (len(other.node_ids) + len(intersection))
                        )
                    )

        return results

    def _prioritize_merged_nodes(
        self, all_hit_nodes: list[NodeWithScore], node_selection: NodeSelection
    ) -> list[NodeWithScore]:
        """Rearrange nodes to prioritize those that survived merging.

        Args:
            all_hit_nodes: All available hit nodes
            node_selection: Current node selection

        Returns:
            list[NodeWithScore]: Rearranged hit nodes
        """
        return [
            node
            for i, node in enumerate(node_selection.hit_nodes)
            if i < len(node_selection.subtrees) and node_selection.subtrees[i].node_ids
        ]

    # ===========================================================================
    # STEP 2 METHODS: Processing expanded nodes
    # ===========================================================================
    def _process_expanded_nodes(self, node_selection: NodeSelection) -> ProcessedNodes:
        """Process expanded nodes by loading them from storage.

        Args:
            node_selection: Selected nodes and their expansions

        Returns:
            ProcessedNodes: Processed node data
        """
        # Prepare the list of all nodes to load
        active_subtrees = [s for s in node_selection.subtrees if s.node_ids]
        all_nodes_to_load = {
            id_ for subtree in active_subtrees for id_ in subtree.node_ids
        }

        logger.debug(f"Loading {len(all_nodes_to_load)} expanded nodes")

        # Load all nodes
        loaded_nodes = {
            n.id_: cast(TreeNode, n)
            for n in self.node_component.get_nodes(
                collection=self.collection,
                node_ids=list(all_nodes_to_load),
                limit=len(all_nodes_to_load),
            )
        }

        # Filter out hit nodes/root nodes with empty subtrees
        filtered_items = []
        for i, (hit_node, root_node) in enumerate(
            zip(node_selection.hit_nodes, node_selection.root_nodes, strict=False)
        ):
            if i < len(node_selection.subtrees) and node_selection.subtrees[i].node_ids:
                filtered_items.append(
                    (hit_node, root_node, node_selection.subtrees[i].node_ids)
                )

        return ProcessedNodes(filtered_items=filtered_items, loaded_nodes=loaded_nodes)

    # ===========================================================================
    # STEP 3 METHODS: Generating final result nodes
    # ===========================================================================
    def _generate_result_nodes(
        self, processed_nodes: ProcessedNodes
    ) -> list[NodeWithScore]:
        """Generate final result nodes by processing loaded nodes.

        Args:
            processed_nodes: Output from _process_expanded_nodes

        Returns:
            list[NodeWithScore]: Final result nodes
        """
        result_nodes: list[NodeWithScore] = []
        logger.debug(
            f"Processing {len(processed_nodes.filtered_items)} items for final results"
        )

        with ThreadPoolExecutor() as executor:
            futures: list[Future[list[NodeWithScore]]] = [
                executor.submit(
                    self._process_node,
                    hit_node,
                    root_node,
                    subtree,
                    processed_nodes.loaded_nodes,
                )
                for hit_node, root_node, subtree in processed_nodes.filtered_items
            ]

            for future in as_completed(futures):
                result_nodes.extend(future.result())

        # Sort by score
        return sorted(result_nodes, key=lambda x: float(x.score or 0), reverse=True)

    def _process_node(
        self,
        hit_node: NodeWithScore,
        root_node: DocumentRootNode,
        current_subtree: set[str],
        loaded_nodes: dict[str, TreeNode],
    ) -> list[NodeWithScore]:
        if not current_subtree or len(current_subtree) == 0:
            # This tree was merged with another
            return []

        # Sort the nodes by their absolute index
        sorted_nodes = sorted(
            [loaded_nodes[node_id] for node_id in current_subtree],
            key=lambda x: x.abs_idx,
        )

        # Post-process the nodes to remove irrelevant content and add diffs
        final_nodes = [n for n in self._rebuild_tree(root_node, sorted_nodes) if n]

        # Prune the content of the final node
        final_nodes = [n for n in self._prune_content(final_nodes) if n]

        # Split the final node into smaller subtrees
        final_nodes = [n for n in self._split_subtrees(final_nodes) if n]

        # Update metadata for the final node
        final_nodes = [self._update_node(hit_node.node, n) for n in final_nodes if n]

        # Frozen and deduplicate the final nodes
        final_nodes = self._deduplicate_subtrees(final_nodes)

        return [
            NodeWithScore(
                node=final_node,
                score=hit_node.score,
            )
            for final_node in final_nodes
            if final_node
        ]

    def _expand(
        self,
        hit_node: TreeNode,
        token_limit: int,
    ) -> ExpansionResult:
        """Expand a node into a subtree with token count.

        Args:
            hit_node: Node to expand
            token_limit: Token limit for expansion

        Returns:
            ExpansionResult: Result of expansion with node IDs and token count
        """
        alg: DocumentTreeExpander = DocumentTreeExpander(
            hit_node,
            token_limit,
        )
        unsorted_node_ids, token_count = alg.fill_window()
        return ExpansionResult(node_ids=unsorted_node_ids, token_count=token_count)

    def _rebuild_tree(
        self, root_node: DocumentRootNode, nodes_to_load: list[TreeNode]
    ) -> list[TreeNode]:
        return TreeNode.rebuild_tree(nodes_to_load, root_node=root_node)

    def _prune_content(self, nodes: list[TreeNode]) -> list[TreeNode]:
        result = [node.prune() for node in nodes]
        return [node for node in result if node]

    def _split_subtrees(self, nodes: list[TreeNode]) -> list[TreeNode]:
        alg: SplitSubtreeAlg = SplitSubtreeAlg()
        subtrees = [alg.split_subtree(node) for node in nodes if node]
        return [subtree for subtrees in subtrees for subtree in subtrees]

    def _deduplicate_subtrees(self, nodes: list[TreeNode]) -> list[TreeNode]:
        # Convert into frozen elements
        frozen_nodes: list[TreeNode] = []
        for node in nodes:
            frozen_nodes.append(FrozenNode.from_node(node))

        # Deduplicate based on the content
        already_seen: set[str] = set()
        deduplicated_nodes: list[TreeNode] = []
        for node in frozen_nodes:
            content = node.get_content(MetadataMode.LLM)
            if content not in already_seen:
                deduplicated_nodes.append(node)
                already_seen.add(content)

        return deduplicated_nodes

    def _update_node(self, hit_node: BaseNode, node: TreeNode) -> TreeNode:
        def regenerate_id(tree_node: TreeNode) -> TreeNode:
            """Regenerate the id of the tree node."""
            content = tree_node.get_content(MetadataMode.LLM)
            tree_node.id_ = str(generate_deterministic_uuid_from_seed(content))
            return tree_node

        def update_metadata(tree_node: TreeNode) -> TreeNode:
            """Update metadata for the tree node."""
            token_count = tree_node.metadata.get(MetadataNode.TOKEN_COUNT.value)
            tree_node.metadata.update(hit_node.metadata)
            tree_node.metadata.update({MetadataNode.TOKEN_COUNT.value: token_count})
            tree_node.excluded_llm_metadata_keys = hit_node.excluded_llm_metadata_keys
            tree_node.excluded_embed_metadata_keys = (
                hit_node.excluded_embed_metadata_keys
            )
            return tree_node

        def update_page(tree_node: TreeNode) -> TreeNode:
            """If page metadata exists, update to use the minimum page number."""
            pages: list[int] = [
                n.metadata.get(MetadataChunk.PAGE.value, -1)
                for n in tree_node.flatten()
                if not isinstance(n, PartialNode)
            ]
            pages = [page for page in pages if page > 0]
            if pages:
                tree_node.metadata[MetadataChunk.PAGE.value] = min(pages)
                if MetadataChunk.PAGE.value not in tree_node.excluded_llm_metadata_keys:
                    tree_node.excluded_llm_metadata_keys.append(
                        MetadataChunk.PAGE.value
                    )
                if (
                    MetadataChunk.PAGE.value
                    not in tree_node.excluded_embed_metadata_keys
                ):
                    tree_node.excluded_embed_metadata_keys.append(
                        MetadataChunk.PAGE.value
                    )
            return tree_node

        node = regenerate_id(node)
        node = update_metadata(node)
        node = update_page(node)
        return node
