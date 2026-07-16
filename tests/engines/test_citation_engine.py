from unittest.mock import Mock

import pytest
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.llms.types import (
    ChatMessage,
    MessageRole,
)
from llama_index.core.schema import MetadataMode, NodeWithScore

from private_gpt.components.engines.citations.format import format_context
from private_gpt.components.engines.citations.types import Document
from private_gpt.components.engines.citations.utils import (
    ORIGINAL_END_TOKEN,
    ORIGINAL_START_TOKEN,
    SHORTER_ID_FIELD,
    SHORTER_ID_LENGTH,
    extract_citations_by_original_text,
    format_cite,
    replace_citations_in_text,
)
from private_gpt.components.ingest.metadata_helper import MetadataKeys
from private_gpt.components.readers.nodes.text_node import TextNode
from private_gpt.components.readers.nodes.text_node import TextNode as Node

# Token variations for testing
TOKEN_VARIATIONS: list[tuple[str, str]] = [
    (ORIGINAL_START_TOKEN, ORIGINAL_END_TOKEN),
    ("[", "]"),
    ("{", "}"),
    ("\\(", ")\\"),
]

# ID length variations
ID_LENGTHS: list[int] = [
    SHORTER_ID_LENGTH,
    2,
    4,
    8,
]

# Generate all combinations of token variations and ID lengths
PARAMS: list[tuple[str, str, int]] = [
    (start_token, end_token, id_length)
    for start_token, end_token in TOKEN_VARIATIONS
    for id_length in ID_LENGTHS
]

# Deduplicate the parameters
PARAMS = list(set(PARAMS))


def random_id_with_length(length: int) -> str:
    from random import choice
    from string import ascii_uppercase

    return "".join(choice(ascii_uppercase) for i in range(length))


@pytest.fixture
def mock_retriever() -> Mock:
    return Mock(spec=BaseRetriever)


def convert_nodes_to_documents_list(
    nodes: list[NodeWithScore],
) -> list[Document]:
    """Convert nodes to documents list."""
    return [
        Document(
            type="document",
            id_=node.node_id,
            shorter_id=node.metadata.get(SHORTER_ID_FIELD),
            document_id=node.metadata.get(MetadataKeys.ARTIFACT_ID.value),
            text=node.get_content(metadata_mode=MetadataMode.LLM),
        )
        for node in nodes or []
    ]


@pytest.mark.parametrize(("start_token", "end_token", "id_length"), PARAMS)
def test_extract_citations_empty_text(
    mock_retriever: Mock, start_token: str, end_token: str, id_length: int
) -> None:
    id_1 = random_id_with_length(id_length)
    mock_retriever.retrieve.return_value = [
        NodeWithScore(
            node=TextNode(
                id_=id_1,
                text="Lorem ipsum dolor sit amet.",
                extra_info={MetadataKeys.ARTIFACT_ID.value: "1"},
            ),
            score=1.0,
        ),
    ]
    documents = convert_nodes_to_documents_list(mock_retriever.retrieve())

    response_mock = ""
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == response_mock
    assert citations == []


@pytest.mark.parametrize(("start_token", "end_token", "id_length"), PARAMS)
def test_extract_citations_without_citations(
    mock_retriever: Mock, start_token: str, end_token: str, id_length: int
) -> None:
    id_1 = random_id_with_length(id_length)
    mock_retriever.retrieve.return_value = [
        NodeWithScore(
            node=TextNode(
                id_=id_1,
                text="Lorem ipsum dolor sit amet.",
                extra_info={MetadataKeys.ARTIFACT_ID.value: "1"},
            ),
            score=1.0,
        ),
    ]
    documents = convert_nodes_to_documents_list(mock_retriever.retrieve())

    # Case 1: No citations in the text
    response_mock = "Lorem ipsum dolor sit amet."
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == response_mock
    assert citations == []

    # Case 2: No citations but it seems like there is one
    random_id = random_id_with_length(id_length + 1)
    response_mock = f"Lorem ipsum dolor sit amet {start_token}"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == "Lorem ipsum dolor sit amet "
    assert citations == []

    response_mock = f"Lorem ipsum dolor sit amet {start_token}{random_id}"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == "Lorem ipsum dolor sit amet "
    assert citations == []

    response_mock = f"Lorem ipsum dolor sit amet {start_token}{random_id}{end_token}"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == response_mock
    assert citations == []


@pytest.mark.parametrize(("start_token", "end_token", "id_length"), PARAMS)
def test_extract_citations_with_uncompleted_citations(
    mock_retriever: Mock, start_token: str, end_token: str, id_length: int
) -> None:
    id_1 = random_id_with_length(id_length)
    mock_retriever.retrieve.return_value = [
        NodeWithScore(
            node=TextNode(
                id_=id_1,
                text="Lorem ipsum dolor sit amet.",
                extra_info={MetadataKeys.ARTIFACT_ID.value: "1"},
            ),
            score=1.0,
        ),
    ]
    documents = convert_nodes_to_documents_list(mock_retriever.retrieve())

    # Case 1: Completed start token but uncompleted value
    response_mock = f"Lorem ipsum dolor sit amet. {start_token}"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == "Lorem ipsum dolor sit amet. "
    assert citations == []

    # Case 2: Completed start token but uncompleted end token
    i = 1
    current_generated_id = id_1[0]
    while i < id_length - 1:
        response_mock = (
            f"Lorem ipsum dolor sit amet. {start_token}{current_generated_id}"
        )
        formatted_response, citations, _ = extract_citations_by_original_text(
            response_mock,
            documents,
            start_token,
            end_token,
            shorter_id_length=id_length,
        )
        assert formatted_response == "Lorem ipsum dolor sit amet. "
        assert citations == []
        i += 1
        current_generated_id += id_1[i]

    # Case 3: Completed start token but uncompleted end token
    if (
        len(end_token) > 1
    ):  # Only run this test if end_token has more than one character
        response_mock = f"Lorem ipsum dolor sit amet. {start_token}{id_1}"
        formatted_response, citations, _ = extract_citations_by_original_text(
            response_mock,
            documents,
            start_token,
            end_token,
            shorter_id_length=id_length,
        )
        assert formatted_response == "Lorem ipsum dolor sit amet. "
        assert citations == []

        response_mock = (
            f"Lorem ipsum dolor sit amet. {start_token}{id_1}{end_token[:1]}"
        )
        formatted_response, citations, _ = extract_citations_by_original_text(
            response_mock,
            documents,
            start_token,
            end_token,
            shorter_id_length=id_length,
        )
        assert formatted_response == "Lorem ipsum dolor sit amet. "
        assert citations == []

        response_mock = f"Lorem ipsum dolor sit amet. {start_token}{id_1}{end_token}"
        formatted_response, citations, _ = extract_citations_by_original_text(
            response_mock,
            documents,
            start_token,
            end_token,
            shorter_id_length=id_length,
        )
        assert (
            formatted_response
            == f"Lorem ipsum dolor sit amet. {format_cite(0, documents[0], 0)}"
        )
        assert len(citations) == 1

    # Case 4: Multiples cites in the same text
    response_mock = f"Lorem ipsum dolor sit amet. {start_token}{id_1}{end_token} {start_token}{id_1}{end_token}"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert (
        formatted_response
        == f"Lorem ipsum dolor sit amet. {format_cite(0, documents[0], 0)} {format_cite(0, documents[0], 0)}"
    )
    assert len(citations) == 2

    response_mock = (
        f"Lorem ipsum dolor sit amet. {start_token}{id_1}, {id_1}{end_token}"
    )
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert (
        formatted_response
        == f"Lorem ipsum dolor sit amet. {format_cite(0, documents[0], 0)},{format_cite(0, documents[0], 0)}"
    )
    assert len(citations) == 2


@pytest.mark.parametrize(("start_token", "end_token", "id_length"), PARAMS)
def test_extract_citations_with_citations(
    mock_retriever: Mock, start_token: str, end_token: str, id_length: int
) -> None:
    id_1 = random_id_with_length(id_length)
    mock_retriever.retrieve.return_value = [
        NodeWithScore(
            node=TextNode(
                id_=id_1,
                text="Lorem ipsum dolor sit amet.",
                extra_info={MetadataKeys.ARTIFACT_ID.value: "1"},
            ),
            score=1.0,
        ),
    ]
    documents = convert_nodes_to_documents_list(mock_retriever.retrieve())

    response_mock = f"Lorem ipsum dolor sit amet. {start_token}{id_1}{end_token}"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert (
        formatted_response
        == f"Lorem ipsum dolor sit amet. {format_cite(0, documents[0], 0)}"
    )
    assert len(citations) == 1


@pytest.mark.parametrize(("start_token", "end_token", "id_length"), PARAMS)
def test_extract_citations_with_citations_and_quotes(
    mock_retriever: Mock, start_token: str, end_token: str, id_length: int
) -> None:
    id_1 = random_id_with_length(id_length)
    mock_retriever.retrieve.return_value = [
        NodeWithScore(
            node=TextNode(
                id_=id_1,
                text="Lorem ipsum dolor sit amet.",
                extra_info={MetadataKeys.ARTIFACT_ID.value: "1"},
            ),
            score=1.0,
        ),
    ]
    documents = convert_nodes_to_documents_list(mock_retriever.retrieve())

    # Case 1.1. Streaming a non a ciations
    response_mock = "Lorem ipsum dolor sit amet. `"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == "Lorem ipsum dolor sit amet. "
    assert len(citations) == 0

    # Case 1.2. Streaming a non a ciations
    response_mock = "Lorem ipsum dolor sit amet. `just"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == "Lorem ipsum dolor sit amet. `just"

    assert len(citations) == 0

    # Case 1.3. Streaming a non a ciations
    response_mock = "Lorem ipsum dolor sit amet. `just a quote`"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == "Lorem ipsum dolor sit amet. `just a quote`"

    # Case 2.1. Initial streaming citation without citation
    response_mock = "Lorem ipsum dolor sit amet. `"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == "Lorem ipsum dolor sit amet. "
    assert len(citations) == 0

    # Case 2.2. Initial streaming citation without citation
    response_mock = f"Lorem ipsum dolor sit amet. `{start_token}{id_1}"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == "Lorem ipsum dolor sit amet. "
    assert len(citations) == 0

    # Case 2.3. Initial streaming citation with citation
    response_mock = f"Lorem ipsum dolor sit amet. `{start_token}{id_1}{end_token}"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert (
        formatted_response
        == f"Lorem ipsum dolor sit amet. {format_cite(0, documents[0], 0)}"
    )
    assert len(citations) == 1

    # Case 2.4. Initial streaming citation with citation + quote
    response_mock = f"Lorem ipsum dolor sit amet. `{start_token}{id_1}{end_token}`"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert (
        formatted_response
        == f"Lorem ipsum dolor sit amet. {format_cite(0, documents[0], 0)}"
    )
    assert len(citations) == 1

    # Case 3.1. Code block (streaming with ```)
    response_mock = """\
    ### Bubble Sort Algorithm
    #### Pseudocode Implementation

    ```markdown
    # Function to perform bubble sort on an array
    """
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == response_mock
    assert len(citations) == 0

    # Case 3.2. Code block (starting with ```)
    response_mock = """\
        ### Bubble Sort Algorithm
        #### Pseudocode Implementation

        ```markdown
        # Function to perform bubble sort on an array
        BubbleSort(arr):
            n = length of arr

            # Repeat through each element in the list until it becomes sorted
            For i From 0 To n - 1:
                swapped = False
        ```
        """
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == response_mock
    assert len(citations) == 0

    # Case 3.3. Code block (starting with ``` without ending)
    response_mock = """\
        ### Bubble Sort Algorithm
        #### Pseudocode Implementation

        ```markdown
        # Function to perform bubble sort on an array
        BubbleSort(arr):
            n = length of arr

            # Repeat through each element in the list until it becomes sorted
            For i From 0 To n - 1:
                swapped = False

                # Compare adjacent elements and swap if necessary
                For j From 0 To n - i - 1:
                    If arr[j] > arr[j + 1]:
                        Swap arr[j] And arr[j + 1]
                        swapped = True

                # If no swaps were made during pass, then the list is already sorted
                If Not swapped:
                    Break

            Return arr
        """
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == response_mock
    assert len(citations) == 0

    # Case 3.4. Code block
    response_mock = """\
        ### Bubble Sort Algorithm
        #### Pseudocode Implementation

        ```markdown
        # Function to perform bubble sort on an array
        BubbleSort(arr):
            n = length of arr

            # Repeat through each element in the list until it becomes sorted
            For i From 0 To n - 1:
                swapped = False

                # Compare adjacent elements and swap if necessary
                For j From 0 To n - i - 1:
                    If arr[j] > arr[j + 1]:
                        Swap arr[j] And arr[j + 1]
                        swapped = True

                # If no swaps were made during pass, then the list is already sorted
                If Not swapped:
                    Break

            Return arr
        ```
        """
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == response_mock
    assert len(citations) == 0


@pytest.mark.parametrize(("start_token", "end_token", "id_length"), PARAMS)
def test_unknown_citations_in_history(
    mock_retriever: Mock, start_token: str, end_token: str, id_length: int
) -> None:
    id_1 = random_id_with_length(id_length)
    id_2 = random_id_with_length(id_length)
    mock_retriever.retrieve.return_value = [
        NodeWithScore(
            node=TextNode(
                id_=id_1,
                text="Lorem ipsum dolor sit amet.",
                extra_info={MetadataKeys.ARTIFACT_ID.value: "1"},
            ),
            score=1.0,
        ),
        NodeWithScore(
            node=TextNode(
                id_=id_2,
                text="Lorem ipsum dolor sit amet.",
                extra_info={MetadataKeys.ARTIFACT_ID.value: "2"},
            ),
            score=1.0,
        ),
    ]
    documents = convert_nodes_to_documents_list(mock_retriever.retrieve())
    chat_history = [
        ChatMessage(
            content="Lorem ipsum dolor sit amet.",
            role=MessageRole.USER,
        ),
        ChatMessage(
            content=f"Lorem ipsum dolor sit amet. {format_cite(1, documents[0], 0)}",
            role=MessageRole.ASSISTANT,
        ),
    ]
    unk_token = "UNK"
    replaced_chat_history = replace_citations_in_text(
        chat_history, documents[1:], unk_token
    )
    assert replaced_chat_history[1].content == "Lorem ipsum dolor sit amet. "


@pytest.mark.parametrize(("start_token", "end_token", "id_length"), PARAMS)
def test_success_citations_in_history(
    mock_retriever: Mock, start_token: str, end_token: str, id_length: int
) -> None:
    id_1 = random_id_with_length(id_length)
    id_2 = random_id_with_length(id_length)
    mock_retriever.retrieve.return_value = [
        NodeWithScore(
            node=TextNode(
                id_=id_1,
                text="Lorem ipsum dolor sit amet.",
                extra_info={MetadataKeys.ARTIFACT_ID.value: "1"},
            ),
            score=1.0,
        ),
        NodeWithScore(
            node=TextNode(
                id_=id_2,
                text="Lorem ipsum dolor sit amet.",
                extra_info={MetadataKeys.ARTIFACT_ID.value: "2"},
            ),
            score=1.0,
        ),
    ]
    documents = convert_nodes_to_documents_list(mock_retriever.retrieve())
    chat_history = [
        ChatMessage(
            content="Lorem ipsum dolor sit amet.",
            role=MessageRole.USER,
        ),
        ChatMessage(
            content=f"Lorem ipsum dolor sit amet. {format_cite(1, documents[0], 0)}",
            role=MessageRole.ASSISTANT,
        ),
    ]
    unk_token = "UNK"

    replaced_chat_history = replace_citations_in_text(
        chat_history, documents, unk_token, start_token, end_token
    )
    assert (
        replaced_chat_history[1].content
        == f"Lorem ipsum dolor sit amet. {start_token}{id_1}{end_token}"
    )


def mock_tokenizer_fn(text: str) -> list[str]:
    return text.split()


def test_generates_context_within_token_limit() -> None:
    nodes = [
        NodeWithScore(
            node=TextNode(
                id_="1",
                text="Content 1",
                abs_idx=1,
                extra_info={MetadataKeys.FILENAME.value: "doc1.pdf"},
            ),
            score=0.9,
        ),
        NodeWithScore(
            node=TextNode(
                id_="2",
                text="Content 2",
                abs_idx=2,
                extra_info={MetadataKeys.FILENAME.value: "doc1.pdf"},
            ),
            score=0.8,
        ),
        NodeWithScore(
            node=TextNode(
                id_="3",
                text="Content 3",
                abs_idx=1,
                extra_info={MetadataKeys.FILENAME.value: "doc2.pdf"},
            ),
            score=0.95,  # Higher score than node 1
        ),
    ]

    potential_nodes = nodes[2:]  # Nodes 3 and 4
    token_limit = len(mock_tokenizer_fn(format_context(nodes=potential_nodes)[1]))
    limited_nodes, context = format_context(
        nodes=nodes, token_limit=token_limit, tokenizer_fn=mock_tokenizer_fn
    )

    # Should choose the highest scoring node
    assert len(limited_nodes) == 1
    assert limited_nodes[0].id_ == "3"

    # Check content
    assert "Content 3" in context
    assert "Content 1" not in context
    assert "Content 2" not in context


def test_generates_context_within_large_token_limit() -> None:
    """Test that the function includes all nodes when token limit allows."""
    nodes = [
        NodeWithScore(
            node=TextNode(
                id_="1",
                text="Content 1",
                abs_idx=1,
                extra_info={MetadataKeys.FILENAME.value: "doc1.pdf"},
            ),
            score=0.8,
        ),
        NodeWithScore(
            node=TextNode(
                id_="2",
                text="Content 2",
                abs_idx=2,
                extra_info={MetadataKeys.FILENAME.value: "doc1.pdf"},
            ),
            score=0.7,
        ),
        NodeWithScore(
            node=TextNode(
                id_="3",
                text="Content 3",
                abs_idx=1,
                extra_info={MetadataKeys.FILENAME.value: "doc2.pdf"},
            ),
            score=0.9,
        ),
    ]

    token_limit = 1000  # Large enough to include all nodes
    limited_nodes, context = format_context(
        nodes=nodes, token_limit=token_limit, tokenizer_fn=mock_tokenizer_fn
    )

    # Should include all nodes
    assert len(limited_nodes) == 3

    # Check document grouping
    assert "doc1.pdf" in context
    assert "doc2.pdf" in context

    # Check content
    assert "Content 1" in context
    assert "Content 2" in context
    assert "Content 3" in context


def test_prioritizes_by_score_within_token_limit() -> None:
    """Test that nodes are prioritized by score when token limit is restrictive."""
    nodes = [
        NodeWithScore(
            node=Node(
                id_="1",
                text="Low priority content",
                abs_idx=1,
                extra_info={MetadataKeys.FILENAME.value: "doc1.pdf"},
            ),
            score=0.5,
        ),
        NodeWithScore(
            node=Node(
                id_="2",
                text="Medium priority content",
                abs_idx=2,
                extra_info={MetadataKeys.FILENAME.value: "doc1.pdf"},
            ),
            score=0.7,
        ),
        NodeWithScore(
            node=Node(
                id_="3",
                text="High priority content",
                abs_idx=3,
                extra_info={MetadataKeys.FILENAME.value: "doc1.pdf"},
            ),
            score=0.9,
        ),
        NodeWithScore(
            node=Node(
                id_="4",
                text="Highest priority content",
                abs_idx=1,
                extra_info={MetadataKeys.FILENAME.value: "doc2.pdf"},
            ),
            score=0.95,
        ),
    ]

    # Set token limit to only include two nodes
    potential_nodes = nodes[2:]  # Nodes 3 and 4
    token_limit = len(mock_tokenizer_fn(format_context(nodes=potential_nodes)[1]))
    limited_nodes, _ = format_context(
        nodes=nodes, token_limit=token_limit, tokenizer_fn=mock_tokenizer_fn
    )

    # Should prioritize highest scoring nodes
    assert len(limited_nodes) == 2
    node_ids = [node.id_ for node in limited_nodes]
    assert "4" in node_ids  # Highest priority
    assert "3" in node_ids  # Second highest


def test_multiple_documents_with_section_indices() -> None:
    nodes = [
        NodeWithScore(
            node=Node(
                id_="1",
                text="Doc1 content",
                abs_idx=5,
                extra_info={MetadataKeys.FILENAME.value: "doc1.pdf"},
            ),
            score=0.8,
        ),
        NodeWithScore(
            node=Node(
                id_="2",
                text="Doc2 content",
                abs_idx=3,
                extra_info={MetadataKeys.FILENAME.value: "doc2.pdf"},
            ),
            score=0.7,
        ),
        NodeWithScore(
            node=Node(
                id_="3",
                text="Doc3 content",
                abs_idx=1,
                extra_info={MetadataKeys.FILENAME.value: "doc3.pdf"},
            ),
            score=0.9,
        ),
    ]

    _, context = format_context(nodes=nodes, tokenizer_fn=mock_tokenizer_fn)

    # Check section formatting
    assert "[1]" in context
    assert "[2]" in context
    assert "[3]" in context

    # Check content
    assert "Doc1 content" in context
    assert "Doc2 content" in context
    assert "Doc3 content" in context


@pytest.mark.parametrize(("start_token", "end_token", "id_length"), PARAMS)
def test_extract_citations_with_inline_code_preceding_citation(
    mock_retriever: Mock, start_token: str, end_token: str, id_length: int
) -> None:
    id_1 = random_id_with_length(id_length)
    id_2 = random_id_with_length(id_length)
    mock_retriever.retrieve.return_value = [
        NodeWithScore(
            node=TextNode(
                id_=id_1,
                text="AML flagged transaction record.",
                extra_info={MetadataKeys.ARTIFACT_ID.value: "1"},
            ),
            score=1.0,
        ),
        NodeWithScore(
            node=TextNode(
                id_=id_2,
                text="Watchlist compliance record.",
                extra_info={MetadataKeys.ARTIFACT_ID.value: "2"},
            ),
            score=0.9,
        ),
    ]
    documents = convert_nodes_to_documents_list(mock_retriever.retrieve())

    # Case 1: Single inline code block immediately preceding a citation
    response_mock = (
        f"Transaction flagged with `flagged_aml = YES` {start_token}{id_1}{end_token}"
    )
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert (
        formatted_response
        == f"Transaction flagged with `flagged_aml = YES` {format_cite(0, documents[0], 0)}"
    ), "Inline code content must be preserved when followed by a valid citation"
    assert len(citations) == 1

    # Case 2: Inline code block with space before citation (common in markdown lists)
    response_mock = (
        f"Clients marked with `watchlist_flag = YES` {start_token}{id_1}{end_token}"
        f" require enhanced due diligence."
    )
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert "`watchlist_flag = YES`" in formatted_response
    assert "require enhanced due diligence." in formatted_response
    assert len(citations) == 1

    # Case 3: Multiple inline code spans with
    # citations across lines (reproduces the report scenario)
    response_mock = (
        f"1. Transaction flagged with `flagged_aml = YES` {start_token}{id_1}{end_token}.\n"
        f"2. Accounts show `watchlist_flag = YES` {start_token}{id_2}{end_token} status.\n"
        f"3. Statuses of `FAILED` or `PENDING` indicate distress {start_token}{id_1}{end_token}."
    )
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert "`flagged_aml = YES`" in formatted_response
    assert "`watchlist_flag = YES`" in formatted_response
    assert "`FAILED`" in formatted_response
    assert "`PENDING`" in formatted_response
    assert len(citations) == 3

    # Case 4: Citation immediately after closing backtick with no space
    response_mock = (
        f"Status is `SETTLED`{start_token}{id_1}{end_token} for this record."
    )
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert "`SETTLED`" in formatted_response
    assert "for this record." in formatted_response
    assert len(citations) == 1


@pytest.mark.parametrize(("start_token", "end_token", "id_length"), PARAMS)
def test_streaming_partial_citations(
    mock_retriever: Mock, start_token: str, end_token: str, id_length: int
) -> None:
    id_1 = random_id_with_length(id_length)
    id_2 = random_id_with_length(id_length)
    mock_retriever.retrieve.return_value = [
        NodeWithScore(
            node=TextNode(
                id_=id_1,
                text="First source.",
                extra_info={MetadataKeys.ARTIFACT_ID.value: "1"},
            ),
            score=1.0,
        ),
        NodeWithScore(
            node=TextNode(
                id_=id_2,
                text="Second source.",
                extra_info={MetadataKeys.ARTIFACT_ID.value: "2"},
            ),
            score=0.9,
        ),
    ]
    documents = convert_nodes_to_documents_list(mock_retriever.retrieve())
    partial_id = id_1[: max(1, id_length - 1)]

    # Case 1: Stream ends mid-ID —
    # no end_token yet received
    response_mock = f"text {start_token}{partial_id}"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == "text "
    assert len(citations) == 0

    # Case 2: Stream ends after full ID
    # but before end of multi-char end_token
    if len(end_token) > 1:
        response_mock = f"text {start_token}{id_1}{end_token[0]}"
        formatted_response, citations, _ = extract_citations_by_original_text(
            response_mock,
            documents,
            start_token,
            end_token,
            shorter_id_length=id_length,
        )
        assert formatted_response == "text "
        assert len(citations) == 0

    # Case 3: Stream ends within a multi-char
    # start_token (only first char received)
    if len(start_token) > 1:
        response_mock = f"text {start_token[0]}"
        formatted_response, citations, _ = extract_citations_by_original_text(
            response_mock,
            documents,
            start_token,
            end_token,
            shorter_id_length=id_length,
        )
        assert formatted_response == f"text {start_token[0]}"
        assert len(citations) == 0

    # Case 4: Inline code with content
    # before a complete citation (unclosed backtick span)
    response_mock = f"text `val={start_token}{id_1}{end_token}"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == f"text `val={format_cite(0, documents[0], 0)}"
    assert len(citations) == 1

    # Case 5: Inline code with content before
    # an incomplete citation (stream cut mid-ID)
    response_mock = f"text `val={start_token}{partial_id}"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == "text `val="
    assert len(citations) == 0

    # Case 6: First citation resolves,
    # second is cut mid-ID
    response_mock = (
        f"First {start_token}{id_1}{end_token} then {start_token}{partial_id}"
    )
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == f"First {format_cite(0, documents[0], 0)} then "
    assert len(citations) == 1

    # Case 7: Bare backtick immediately
    # before an incomplete citation (stream cut mid-ID)
    response_mock = f"`{start_token}{partial_id}"
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == ""
    assert len(citations) == 0

    # Case 8: Two complete citations where
    # the second is preceded by a bare backtick
    response_mock = (
        f"{start_token}{id_1}{end_token} and `{start_token}{id_2}{end_token}"
    )
    formatted_response, citations, _ = extract_citations_by_original_text(
        response_mock, documents, start_token, end_token, shorter_id_length=id_length
    )
    assert formatted_response == (
        f"{format_cite(0, documents[0], 0)} and {format_cite(1, documents[1], 1)}"
    )
    assert len(citations) == 2
