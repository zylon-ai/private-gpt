from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, TextNode

from private_gpt.components.readers.nodes import TreeNode


class ConvertTreeNodeIntoLlamaIndexNodesPostProcessor(BaseNodePostprocessor):
    """Converts tree nodes into Llama Index nodes."""

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: QueryBundle | None = None
    ) -> list[NodeWithScore]:
        new_nodes: list[NodeWithScore] = []
        for nodes_with_score in nodes:
            node = nodes_with_score.node
            if isinstance(node, TreeNode):
                node = TextNode(
                    text=node.get_content(MetadataMode.LLM),
                    extra_info=node.metadata,
                    excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
                    excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
                )
                new_nodes.append(NodeWithScore(node=node, score=nodes_with_score.score))
            else:
                new_nodes.append(nodes_with_score)

        return new_nodes
