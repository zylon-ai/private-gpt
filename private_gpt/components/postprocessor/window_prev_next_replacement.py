from llama_index.core.bridge.pydantic import Field
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from private_gpt.components.node_store.node_store_component import NodeStoreComponent


class WindowPrevNextReplacementPostProcessor(BaseNodePostprocessor):
    """Post-processor to return the previous or next nodes to the current node.

    This post-processor is useful for generating a window context out of the LLM,
    useful for better user understanding of the context.

    Args:
        docstore (BaseDocumentStore): The document store.
        num_nodes (int): The number of nodes to return (default: 1)
        mode (str): The mode of the post-processor.
            Can be "previous", "next", or "both.
    """

    node_component: NodeStoreComponent
    collection: str
    window_length: int = Field(default=1)
    mode: str = Field(default="next")

    @classmethod
    def class_name(cls) -> str:
        return "WindowPrevNextReplacementPostProcessor"

    def get_sibling_text(
        self,
        node_with_score: NodeWithScore,
        max_length: int,
        forward: bool = True,
    ) -> list[str]:
        current_length = 0
        result_text = []
        current_node = node_with_score.node

        while current_length < max_length:
            explored_node_info = (
                current_node.next_node if forward else current_node.prev_node
            )
            if explored_node_info is None:
                break

            explored_node = self.node_component.get_node(
                self.collection, explored_node_info.node_id
            )
            if explored_node is None:
                break

            node_content = explored_node.get_content()

            if forward:
                # Add the text to the end of the list
                # If the text is too long, we need to truncate it from the end
                result_text.append(node_content[: max_length - current_length])
            else:
                # Add the text to the beginning of the list
                # If the text is too long, we need to truncate it from the start
                result_text.insert(0, node_content[-(max_length - current_length) :])

            current_length += len(result_text[-1]) if forward else len(result_text[0])

            if current_length >= max_length:
                break

            current_node = explored_node

        return result_text

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """Postprocess nodes."""
        for node in nodes:
            if self.mode == "previous" or self.mode == "both":
                node.metadata["previous_texts"] = self.get_sibling_text(
                    node, self.window_length, forward=False
                )
                node.node.excluded_llm_metadata_keys.append("previous_texts")
            if self.mode == "next" or self.mode == "both":
                node.metadata["next_texts"] = self.get_sibling_text(
                    node, self.window_length, forward=True
                )
                node.node.excluded_llm_metadata_keys.append("next_texts")

        return nodes
