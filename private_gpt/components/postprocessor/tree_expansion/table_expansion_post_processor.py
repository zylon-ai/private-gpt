import logging

from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore

from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.readers.nodes.table_node import TableNode, TableRowNode
from private_gpt.components.readers.nodes.tree_node import TreeNode

logger = logging.getLogger(__name__)


class TableExpansionPostProcessor(BaseNodePostprocessor):
    node_component: NodeStoreComponent
    collection: str

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        if not nodes:
            return []

        expanded_nodes = []
        table_root_ids = set()

        parent_to_children: dict[str, list[NodeWithScore]] = {}
        for node in nodes:
            if not isinstance(node.node, TreeNode):
                expanded_nodes.append(node)
                continue

            tree_node = node.node
            if isinstance(tree_node, TableRowNode) and tree_node.parent_id:
                parent_id = tree_node.parent_id
                table_root_ids.add(parent_id)

                if parent_id not in parent_to_children:
                    parent_to_children[parent_id] = []

                parent_to_children[parent_id].append(node)
            elif isinstance(tree_node, TableNode):
                table_root_ids.add(tree_node.id_)
                expanded_nodes.append(node)

        if not table_root_ids:
            return nodes

        table_nodes = self.node_component.get_nodes(
            collection=self.collection,
            node_ids=list(table_root_ids),
            limit=len(table_root_ids),
        )

        table_node_map = {
            node.id_: node for node in table_nodes if isinstance(node, TableNode)
        }
        for parent_id, child_hits in parent_to_children.items():
            if parent_id in table_node_map:
                parent_node = table_node_map[parent_id]

                scores = [float(hit.score or 0) for hit in child_hits]
                max_score = max(scores) if scores else 0

                expanded_nodes.append(NodeWithScore(node=parent_node, score=max_score))

        results = sorted(
            expanded_nodes, key=lambda x: float(x.score or 0), reverse=True
        )
        results = list({node.id_: node for node in results}.values())

        return results
