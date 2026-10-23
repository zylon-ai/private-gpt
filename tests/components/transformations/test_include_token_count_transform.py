from collections.abc import Iterator

import pandas as pd
import pytest

from private_gpt.components.ingest.metadata_helper import MetadataNode
from private_gpt.components.ingest.transformations.include_token_count_to_nodes_transform import (
    IncludeTokenCountIntoNodesTransform,
)
from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase
from private_gpt.components.readers.nodes.diff_node import DiffNode
from private_gpt.components.readers.nodes.document_node import DocumentRootNode
from private_gpt.components.readers.nodes.image_node import ImageNode
from private_gpt.components.readers.nodes.list_node import ListItemNode, ListNode
from private_gpt.components.readers.nodes.partial_node import PartialNode
from private_gpt.components.readers.nodes.section_node import SectionNode
from private_gpt.components.readers.nodes.table_node import TableNode, TableRowNode
from private_gpt.components.readers.nodes.text_node import TextNode
from private_gpt.components.readers.nodes.tree_node import TreeNode


def mock_metadata() -> dict[str, str]:
    return {
        "key1": "value1",
        "key2": "value2",
    }


def mock_children() -> Iterator[TreeNode]:
    yield SectionNode(
        children=[
            TextNode(
                text="This is a test",
                extra_info=mock_metadata(),
            ),
            TextNode(
                text="This is a test",
            ),
        ],
    )
    yield TextNode(
        text="This is a test",
        extra_info=mock_metadata(),
    )


@pytest.fixture
def transform() -> IncludeTokenCountIntoNodesTransform:
    return IncludeTokenCountIntoNodesTransform.from_defaults(
        tokenizer=lambda x: x.split()
    )


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (
            DocumentRootNode(
                children=list(mock_children()),
                extra_info=mock_metadata(),
                excluded_llm_metadata_keys=list(mock_metadata().keys()),
            ),
            0,
        ),
        (
            TextNode(
                text="This is a test",
                extra_info=mock_metadata(),
                excluded_llm_metadata_keys=list(mock_metadata().keys()),
            ),
            4,
        ),
        (
            SectionNode(
                text="This is a test",
                children=list(mock_children()),
                extra_info=mock_metadata(),
                excluded_llm_metadata_keys=list(mock_metadata().keys()),
            ),
            4,
        ),
        (
            ImageNode(
                alt_text="This is a test",
                image="#",
                extra_info=mock_metadata(),
                excluded_llm_metadata_keys=list(mock_metadata().keys()),
            ),
            4,
        ),
        (
            ListNode(
                children=list(mock_children()),
                extra_info=mock_metadata(),
                excluded_llm_metadata_keys=list(mock_metadata().keys()),
            ),
            0,
        ),
        (
            ListItemNode(
                text="This is a test",
                extra_info=mock_metadata(),
                excluded_llm_metadata_keys=list(mock_metadata().keys()),
            ),
            4,
        ),
        (
            TableNode(
                df=pd.DataFrame(
                    data={
                        "Column 1": ["Value 1", "Value 2"],
                        "Column 2": ["Value 3", "Value 4"],
                    }
                ),
            ),
            0,
        ),
        (
            TableRowNode(
                header=["Column 1", "Column 2"],
                content=["Value 1", "Value 2"],
                extra_info=mock_metadata(),
                excluded_llm_metadata_keys=list(mock_metadata().keys()),
            ),
            7,
        ),
        (
            PartialNode(type="test", extra_info=mock_metadata()),
            0,
        ),
        (
            DiffNode(
                text="This is a test",
                extra_info=mock_metadata(),
                excluded_llm_metadata_keys=list(mock_metadata().keys()),
            ),
            4,
        ),
    ],
)
def test_token_count(
    transform: IncludeTokenCountIntoNodesTransform,
    node: TreeNode,
    expected: int,
) -> None:
    transformed_nodes = transform([node])
    token_count = transformed_nodes[0].metadata.get(MetadataNode.TOKEN_COUNT.value) or 0
    assert token_count == expected
    if expected:
        assert (
            MetadataNode.TOKEN_COUNT.value
            in transformed_nodes[0].excluded_llm_metadata_keys
        )
        assert (
            MetadataNode.TOKEN_COUNT.value
            in transformed_nodes[0].excluded_embed_metadata_keys
        )


class _CountingTokenizer(TokenizerBase):
    """Records the batches it is asked to tokenize."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def encode_batch(self, texts, add_special_tokens=None):  # type: ignore[no-untyped-def]
        self.batches.append(list(texts))
        return [t.split() for t in texts]


# Only the batch surface is exercised; the rest of the ABC isn't needed.
_CountingTokenizer.__abstractmethods__ = frozenset()


def _nodes() -> list[TextNode]:
    return [
        TextNode(text="one two three"),
        TextNode(text="one two three"),
        TextNode(text="four"),
        TextNode(text=""),
    ]


def test_sync_batch_tokenizes_unique_contents_once() -> None:
    tokenizer = _CountingTokenizer()
    transform = IncludeTokenCountIntoNodesTransform(
        tokenizer=None, batch_tokenizer=tokenizer
    )
    nodes = transform(_nodes())

    assert tokenizer.batches == [["one two three", "four"]]
    counts = [n.metadata.get(MetadataNode.TOKEN_COUNT.value) for n in nodes]
    assert counts == [3, 3, 1, None]


def test_async_batch_uses_aencode_batch_with_bounded_concurrency() -> None:
    import asyncio

    from private_gpt.components.llm.tokenizers.tokenizer_base import (
        AsyncTokenizerBase,
    )

    class _Async(AsyncTokenizerBase):
        in_flight = 0
        max_in_flight = 0

        async def aencode(self, text, add_special_tokens=None):  # type: ignore[no-untyped-def]
            _Async.in_flight += 1
            _Async.max_in_flight = max(_Async.max_in_flight, _Async.in_flight)
            await asyncio.sleep(0)
            _Async.in_flight -= 1
            return text.split()

    # Only the async surface is exercised; the rest of the ABC isn't needed.
    _Async.__abstractmethods__ = frozenset()
    tokenizer = _Async()
    texts = [f"t{i} x" for i in range(100)]

    ids = asyncio.run(tokenizer.aencode_batch(texts, concurrency=4))
    assert [len(i) for i in ids] == [2] * 100
    assert _Async.max_in_flight <= 4

    transform = IncludeTokenCountIntoNodesTransform(
        tokenizer=None, batch_tokenizer=tokenizer
    )
    nodes = asyncio.run(transform.acall(_nodes()))
    counts = [n.metadata.get(MetadataNode.TOKEN_COUNT.value) for n in nodes]
    assert counts == [3, 3, 1, None]
