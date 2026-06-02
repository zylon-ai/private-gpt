import uuid

import pytest
from llama_index.core.schema import (
    NodeRelationship,
    NodeWithScore,
    RelatedNodeInfo,
    TextNode,
)

from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.postprocessor.prev_next_replacement import (
    PrevNextReplacementPostProcessor,
)
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.settings.settings import Settings
from tests.fixtures.mock_injector import MockInjector


@pytest.mark.parametrize("mode", ["previous", "next", "both"])
def test_postprocess_nodes_no_relations_return_nodes(mode: str, injector: MockInjector):
    settings = injector.get(Settings)
    embed_dim = settings.vectorstore.embed_dim

    # Create nodes
    nodes = [
        TextNode(text=f"content {i}", embedding=[0.0] * embed_dim) for i in range(3)
    ]
    nodes_with_score = [NodeWithScore(node=node) for node in nodes]

    collection = "test_collection"
    # Save the nodes in the vector store
    vector_component = injector.get(VectorStoreComponent)
    vector_component.vector_store(collection).add(nodes)

    # Create node component
    node_component = injector.get(NodeStoreComponent)

    # Create a PrevNextReplacementPostProcessor instance
    postprocessor = PrevNextReplacementPostProcessor(
        node_component=node_component, collection=collection, num_nodes=2, mode=mode
    )

    # Call the _postprocess_nodes method
    result = postprocessor._postprocess_nodes(nodes_with_score)

    # Ensure the 3 nodes are returned with the original content
    assert all(result[i].node.get_content() == f"content {i}" for i in range(3))


@pytest.mark.parametrize("mode", ["previous", "next", "both"])
def test_postprocess_nodes(mode: str, injector: MockInjector):
    settings = injector.get(Settings)
    embed_dim = settings.vectorstore.embed_dim

    # Create a set of 10 nodes with prev and next relations.
    # Only add prev relation if the item is not the first.
    # Only add next relation if the item is not the last one.
    iter: list[int] = list(range(10))
    nodes = []
    node_ids = [str(uuid.uuid4()) for _ in iter]
    for i in iter:
        relationships = {}
        if i > 0:
            relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=node_ids[i - 1]
            )
        if i < 9:
            relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=node_ids[i + 1]
            )

        nodes.append(
            TextNode(
                id_=node_ids[i],
                text=f"content {i}",
                relationships=relationships,
                embedding=[0.0] * embed_dim,
            )
        )

    nodes_with_score = [NodeWithScore(node=node) for node in nodes]

    collection = "test_collection"
    # Save the nodes in the vector store
    vector_component = injector.get(VectorStoreComponent)
    vector_component.vector_store(collection).add(nodes)

    # Create node component
    node_component = injector.get(NodeStoreComponent)

    # Create a PrevNextReplacementPostProcessor instance
    postprocessor = PrevNextReplacementPostProcessor(
        node_component=node_component, collection=collection, num_nodes=2, mode=mode
    )

    # Test 1: processing all nodes, add only one occurrence of each content
    result = postprocessor._postprocess_nodes(nodes_with_score)
    assert all(result[i].node.get_content() == f"content {i}" for i in range(3))

    # Test 2: processing nodes which relations don't overlap (1, 6)
    result = postprocessor._postprocess_nodes(
        [nodes_with_score[1], nodes_with_score[6]]
    )
    # Evaluate differently based on the mode
    if mode == "previous":
        assert result[0].node.get_content() == "content 0 content 1"
        assert result[1].node.get_content() == "content 4 content 5 content 6"
    elif mode == "next":
        assert result[0].node.get_content() == "content 1 content 2 content 3"
        assert result[1].node.get_content() == "content 6 content 7 content 8"
    elif mode == "both":
        assert result[0].node.get_content() == "content 0 content 1 content 2 content 3"
        assert (
            result[1].node.get_content()
            == "content 4 content 5 content 6 content 7 content 8"
        )

    # Test 3: processing nodes which relations overlap (2, 4, 5)
    result = postprocessor._postprocess_nodes(
        [nodes_with_score[2], nodes_with_score[4], nodes_with_score[5]]
    )
    # Evaluate differently based on the mode
    if mode == "previous":
        assert result[0].node.get_content() == "content 0 content 1 content 2"
        assert result[1].node.get_content() == "content 3 content 4"
        assert result[2].node.get_content() == "content 5"
    elif mode == "next":
        assert result[0].node.get_content() == "content 2 content 3"
        assert result[1].node.get_content() == "content 4"
        assert result[2].node.get_content() == "content 5 content 6 content 7"
    elif mode == "both":
        assert result[0].node.get_content() == "content 0 content 1 content 2 content 3"
        assert result[1].node.get_content() == "content 4"
        assert result[2].node.get_content() == "content 5 content 6 content 7"
