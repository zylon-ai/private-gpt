from llama_index.core.bridge.pydantic import Field
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import (
    MetadataMode,
    NodeRelationship,
    NodeWithScore,
    QueryBundle,
)

from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.readers.nodes.tree_node import TreeNode


class PrevNextReplacementPostProcessor(BaseNodePostprocessor):
    """Enhances node content with previous and next nodes content.

    Replaces the node content by its enhanced version.
    Manages overlaps, making sure there is no duplicated content in the final nodes.

    Args:
        docstore (BaseDocumentStore): The document store.
        num_nodes (int): The number of nodes to return (default: 1)
        mode (str): The mode of the post-processor.
            Can be "previous", "next", or "both.
    """

    node_component: NodeStoreComponent
    collection: str
    num_nodes: int = Field(default=1)
    mode: str = Field(default="next")

    @classmethod
    def class_name(cls) -> str:
        return "PrevNextReplacementPostProcessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Postprocess nodes."""
        # Make sure we don't duplicate content in the final nodes list
        # Initialize the list with the ids of the input nodes
        added_nodes = [node.node_id for node in nodes]

        for node in [n for n in nodes if not isinstance(n.node, TreeNode)]:
            prev_nodes_to_add: dict[str, NodeWithScore] = {}
            next_nodes_to_add: dict[str, NodeWithScore] = {}
            if self.mode == "previous" or self.mode == "both":
                # Extract the prev nodes
                prev_nodes = self.get_backward_nodes(node, self.num_nodes)

                # Curate prev_nodes removing any node which id is already in added_nodes
                prev_nodes_to_add = {
                    k: v for k, v in prev_nodes.items() if k not in added_nodes
                }
                added_nodes.extend(prev_nodes_to_add.keys())

            if self.mode == "next" or self.mode == "both":
                # Extract the next nodes
                next_nodes = self.get_forward_nodes(node, self.num_nodes)

                # Curate next_nodes removing any node which id is already in added_nodes
                # making sure we don't generate any gaps in the final list
                for next_node_id, node_with_score in next_nodes.items():
                    if next_node_id not in added_nodes:
                        next_nodes_to_add[next_node_id] = node_with_score
                        added_nodes.append(next_node_id)
                    elif next_node_id != node.node_id:
                        # Avoid skipping one node and going with the next to
                        # never create context gaps
                        break

            # Generate final content for the node
            prev_text = (
                " ".join(
                    n.get_content(metadata_mode=MetadataMode.NONE)
                    for n in reversed(prev_nodes_to_add.values())
                )
                + " "
                if len(prev_nodes_to_add) > 0
                else ""
            )
            original_text = node.node.get_content(metadata_mode=MetadataMode.NONE)
            next_text = (
                " "
                + " ".join(
                    n.get_content(metadata_mode=MetadataMode.NONE)
                    for n in next_nodes_to_add.values()
                )
                if len(next_nodes_to_add) > 0
                else ""
            )

            node.node.set_content(prev_text + original_text + next_text)

        return nodes

    def get_forward_nodes(
        self, node_with_score: NodeWithScore, num_nodes: int
    ) -> dict[str, NodeWithScore]:
        """Get forward nodes from vector store."""
        node = node_with_score.node
        nodes: dict[str, NodeWithScore] = {node.node_id: node_with_score}
        cur_count = 0
        while cur_count < num_nodes:
            if NodeRelationship.NEXT not in node.relationships:
                break

            next_node_info = node.next_node
            if next_node_info is None:
                break

            next_node_id = next_node_info.node_id
            next_node = self.node_component.get_node(self.collection, next_node_id)
            if next_node is None:
                break
            nodes[next_node.node_id] = NodeWithScore(node=next_node)
            node = next_node
            cur_count += 1
        return nodes

    def get_backward_nodes(
        self, node_with_score: NodeWithScore, num_nodes: int
    ) -> dict[str, NodeWithScore]:
        """Get backward nodes from vector store."""
        node = node_with_score.node
        nodes: dict[str, NodeWithScore] = {node.node_id: node_with_score}
        cur_count = 0
        while cur_count < num_nodes:
            prev_node_info = node.prev_node
            if prev_node_info is None:
                break
            prev_node_id = prev_node_info.node_id
            prev_node = self.node_component.get_node(self.collection, prev_node_id)
            if prev_node is None:
                break
            nodes[prev_node.node_id] = NodeWithScore(node=prev_node)
            node = prev_node
            cur_count += 1
        return nodes
