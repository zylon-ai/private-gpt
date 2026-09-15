from typing import Any

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from pydantic import Field

from private_gpt.artifact_index.base_artifact_index import ExtendIndex, embed_text_for
from private_gpt.components.ingest.transformations.flatten_tree_nodes_transform import (
    FlattenTreeNodesTransform,
)
from private_gpt.components.ingest.transformations.markdown_to_tree_transform import (
    MarkdownTreeNodeParser,
)

FILE_NAME = "DATA-WORKBENCH.md"
HEADER = f"file_name: {FILE_NAME}"

MARKDOWN = """\
# Data Workbench

The workbench exposes DuckDB over SQL.

## Sources

* Parquet snapshots
* CSV exports
"""


class RecordingEmbedding(MockEmbedding):
    """Mock embedding model that remembers the texts it was asked to embed."""

    texts: list[str] = Field(default_factory=list)

    def _get_text_embedding(self, text: str) -> list[float]:
        self.texts.append(text)
        return super()._get_text_embedding(text)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        self.texts.append(text)
        return await super()._aget_text_embedding(text)


def _parsed_nodes() -> list[BaseNode]:
    """Parse MARKDOWN the way the readers do, with the metadata a reader attaches."""
    document = TextNode(
        text=MARKDOWN,
        metadata={"file_name": FILE_NAME, "file_hash": "abc123"},
    )
    trees = MarkdownTreeNodeParser.from_defaults()([document])
    return list(FlattenTreeNodesTransform.from_defaults()(trees))


def _extend_index(**kwargs: Any) -> tuple[ExtendIndex[Any], RecordingEmbedding]:
    embed_model = RecordingEmbedding(embed_dim=8)
    index = VectorStoreIndex(
        nodes=[],
        embed_model=embed_model,
        storage_context=StorageContext.from_defaults(),
    )
    return ExtendIndex(index, embed_size=8, **kwargs), embed_model


def test_tree_parser_leaves_the_file_name_out_of_the_embed_content() -> None:
    # The defect: the name is in the metadata, but not in what the node renders.
    nodes = _parsed_nodes()

    assert nodes
    for node in nodes:
        assert node.metadata["file_name"] == FILE_NAME
        assert FILE_NAME not in node.get_content(metadata_mode=MetadataMode.EMBED)


def test_file_name_is_prepended_once_at_every_level_of_the_tree() -> None:
    nodes = [
        node
        for node in _parsed_nodes()
        if node.get_content(metadata_mode=MetadataMode.EMBED).strip()
    ]

    # Root, sections, text, the list and its items: internal nodes are indexed too,
    # and render their children, so each must still carry exactly one header.
    assert len(nodes) > 3
    for node in nodes:
        text = embed_text_for(node)
        assert text.startswith(HEADER + "\n")
        assert text.count(HEADER) == 1


def test_llm_view_is_unchanged() -> None:
    for node in _parsed_nodes():
        assert FILE_NAME not in node.get_content(metadata_mode=MetadataMode.LLM)


def test_other_metadata_stays_out_of_the_embedding() -> None:
    for node in _parsed_nodes():
        assert "abc123" not in embed_text_for(node)


def test_node_without_content_stays_empty() -> None:
    node = TextNode(
        text="",
        metadata={"file_name": FILE_NAME},
        excluded_embed_metadata_keys=["file_name"],
    )

    assert embed_text_for(node) == ""


def test_node_without_a_file_name_is_unchanged() -> None:
    node = TextNode(text="plain text")

    assert embed_text_for(node) == "plain text"


def test_node_that_already_embeds_its_file_name_gets_no_second_copy() -> None:
    node = TextNode(text="plain text", metadata={"file_name": FILE_NAME})

    assert embed_text_for(node) == node.get_content(metadata_mode=MetadataMode.EMBED)
    assert embed_text_for(node).count(HEADER) == 1


def test_embedding_model_receives_the_file_name() -> None:
    extend_index, embed_model = _extend_index()
    nodes = _parsed_nodes()

    embedded = extend_index._get_node_with_embedding(nodes, show_progress=False)

    assert len(embedded) == len(nodes)
    assert embed_model.texts
    assert all(text.startswith(HEADER + "\n") for text in embed_model.texts)


async def test_embedding_model_receives_the_file_name_async() -> None:
    extend_index, embed_model = _extend_index()
    nodes = _parsed_nodes()

    embedded = await extend_index._aget_node_with_embedding(nodes, show_progress=False)

    assert len(embedded) == len(nodes)
    assert embed_model.texts
    assert all(text.startswith(HEADER + "\n") for text in embed_model.texts)


def test_truncation_keeps_the_file_name() -> None:
    extend_index, embed_model = _extend_index(max_truncate_length=len(HEADER) + 10)

    extend_index._get_node_with_embedding(_parsed_nodes(), show_progress=False)

    assert embed_model.texts
    assert all(text.startswith(HEADER + "\n") for text in embed_model.texts)
