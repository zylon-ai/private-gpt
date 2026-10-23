import pytest
from llama_index.core.node_parser import TokenTextSplitter

# Import the functions and classes from your modules.
from private_gpt.components.ingest.transformations.sentence_tree_node_parser import (
    SentenceTreeNodeParser,
    contains_arabic,
    split_by_sentence_tokenizer,
    split_by_sentence_tokenizer_internal,
)
from private_gpt.components.readers.nodes.chunk_node import ChunkNode
from private_gpt.components.readers.nodes.document_node import DocumentRootNode
from private_gpt.components.readers.nodes.text_node import TextNode


class DummyTextSplitter(TokenTextSplitter):
    """A dummy text splitter that splits text into fixed-size chunks."""

    def split_text(self, text: str) -> list[str]:
        # Splits the text into chunks of maximum length 'chunk_size'
        return [
            text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)
        ]


@pytest.mark.parametrize(
    ("input_text", "expected_chunks", "include_metadata"),
    [
        (
            "This is a single sentence.",
            0,
            True,
        ),  # Single sentence (no splitting required)
        (
            "Sentence one. Sentence two. Sentence three.",
            3,
            True,
        ),  # Multiple sentences; each sentence becomes a chunk.
        ("", 0, True),  # Empty text should yield no chunks.
        ("Sentence with metadata.", 0, True),  # Single sentence, metadata included.
        ("Another test sentence.", 0, False),  # Single sentence, no metadata.
    ],
)
def test_sentence_tree_node_parser(
    input_text: str,
    expected_chunks: int,
    include_metadata: bool,
) -> None:
    # Initialize the parser with default (non-fallback) behavior.
    parser = SentenceTreeNodeParser.from_defaults(
        include_metadata=include_metadata,
    )

    # Create a root DocumentRoot node and a child TextNode.
    root_node = DocumentRootNode(metadata={"root_key": "root_value"})
    text_node = TextNode(
        text=input_text,
        metadata={"child_key": "child_value"} if include_metadata else {},
    )
    root_node.add_child(text_node)

    # Parse the tree.
    parsed_nodes = parser._parse_nodes([root_node])
    assert len(parsed_nodes) == 1  # Root node remains.
    root_result = parsed_nodes[0]
    assert len(root_result.children) == 1
    parsed_child = root_result.children[0]
    assert isinstance(parsed_child, TextNode)
    assert len(parsed_child.children) == expected_chunks

    # Validate metadata and that each chunk's text is a substring of the input.
    for _, chunk in enumerate(parsed_child.children):
        assert isinstance(chunk, ChunkNode)
        assert chunk.text.strip() in input_text
        if include_metadata:
            # The metadata from both the child and root should be present.
            assert chunk.metadata == {
                "child_key": "child_value",
                "root_key": "root_value",
            }
        else:
            assert chunk.metadata == {}


def test_contains_arabic() -> None:
    test_cases: list[tuple[str, bool]] = [
        ("هذا نص عربي", True),
        ("This has some Arabic: مرحبا", True),
        ("This is English text", False),
        ("1234567890", False),
        ("これは日本語です", False),
        ("", False),
        (" ", False),
    ]
    for text, expected in test_cases:
        result: bool = contains_arabic(text)
        assert result == expected


def test_split_by_sentence_tokenizer_function() -> None:
    tokenizer = split_by_sentence_tokenizer()
    assert callable(tokenizer)
    result: list[str] = tokenizer("This is a test.")
    assert isinstance(result, list)


def test_split_arabic_sentences() -> None:
    arabic_text: str = (
        "تتميز دبي ببيئة استثمارية جاذبة. بفضل السياسات الاقتصادية المبتكرة."
    )
    result: list[str] = split_by_sentence_tokenizer_internal(arabic_text)
    assert len(result) == 2
    assert result[0].startswith("تتميز")
    assert result[1].startswith("بفضل")


def test_split_english_sentences() -> None:
    english_text: str = "This is a sentence. This is another sentence."
    result: list[str] = split_by_sentence_tokenizer_internal(english_text)
    assert len(result) == 2
    assert result[0].startswith("This is a")
    assert result[1].startswith("This is another")


def test_complex_arabic_text() -> None:
    text: str = """
   تعتبر اللغة العربية من أقدم اللغات في العالم. وهي لغة القرآن الكريم. تتميز بجمال
    خطها وبلاغتها! هل تعلم أن هناك أكثر من 300 مليون شخص يتحدثون اللغة العربية؟
    """
    result: list[str] = split_by_sentence_tokenizer_internal(text)
    # Expect 4 sentences based on '.', '!', and '؟'
    assert len(result) == 4


def test_edge_cases() -> None:
    test_cases: list[tuple[str, int]] = [
        ("", 0),
        ("hello", 1),
        ("مرحبا", 1),
        ("This is a sentence without punctuation", 1),
        ("هذا نص عربي بدون علامات ترقيم", 1),
        ("Sentence one.   Sentence two.", 2),
    ]
    for text, expected_count in test_cases:
        result: list[str] = split_by_sentence_tokenizer_internal(text)
        assert len(result) == expected_count


@pytest.fixture
def dummy_text_splitter():
    return DummyTextSplitter(
        chunk_size=20,
        chunk_overlap=0,
    )


def test_max_length_chunker_applied(dummy_text_splitter) -> None:
    # Create a long sentence that will exceed the 20-character chunk size.
    long_text = (
        "This is a very long sentence that should be split into multiple chunks."
    )
    # Initialize parser with our dummy fallback text splitter.
    parser = SentenceTreeNodeParser.from_defaults(
        fallback_text_splitter=dummy_text_splitter,
        include_metadata=True,
    )
    # Create a DocumentRoot with a TextNode.
    root_node = DocumentRootNode(metadata={"root_key": "root_value"})
    text_node = TextNode(
        text=long_text,
        metadata={"child_key": "child_value"},
    )
    root_node.add_child(text_node)

    # Parse the node tree.
    parsed_nodes = parser._parse_nodes([root_node])
    root_result = parsed_nodes[0]
    # Get the TextNode child.
    parsed_child = root_result.children[0]
    # The fallback should have split the sentence.
    assert len(parsed_child.children) > 1

    # Check that each chunk is no longer than 10 characters (per dummy splitter)
    for chunk in parsed_child.children:
        assert isinstance(chunk, ChunkNode)
        assert len(chunk.text) <= 20
        # Ensure metadata is combined from both the TextNode and DocumentRoot.
        assert chunk.metadata == {
            "child_key": "child_value",
            "root_key": "root_value",
        }


def test_no_chunking_for_short_text(dummy_text_splitter) -> None:
    short_text = "Short sentence."
    parser = SentenceTreeNodeParser.from_defaults(
        fallback_text_splitter=dummy_text_splitter,
        include_metadata=False,
    )
    root_node = DocumentRootNode(metadata={"root_key": "root_value"})
    text_node = TextNode(text=short_text, metadata={})
    root_node.add_child(text_node)

    parsed_nodes = parser._parse_nodes([root_node])
    root_result = parsed_nodes[0]
    parsed_child = root_result.children[0]
    # If there is only one sentence and it's short, no chunks should be created.
    assert len(parsed_child.children) == 0
    # The text should remain unchanged.
    assert parsed_child.text == short_text
