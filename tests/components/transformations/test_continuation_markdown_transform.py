from private_gpt.components.ingest.transformations.make_continuation_markdown_transform import (
    MakeContinuationMarkdownTransform,
    MarkdownMerger,
)
from private_gpt.components.readers.nodes import TextNode


def test_merge_lists() -> None:
    merger = MarkdownMerger()
    pages = [
        "- Item 1\n- Item 2",
        "- Item 3\n- Item 4",
    ]
    expected = ["- Item 1\n- Item 2\n- Item 3\n- Item 4"]
    assert merger.merge_markdown_documents(pages) == expected


def test_merge_tables() -> None:
    merger = MarkdownMerger()
    # Case 1. One table in two pages
    pages = [
        "| Column1 | Column2 |\n|---------|---------|",
        "| Value1  | Value2  |",
    ]
    expected = ["| Column1 | Column2 |\n|---------|---------|\n| Value1  | Value2  |"]
    assert merger.merge_markdown_documents(pages) == expected

    # Case 2. Two tables with different headers in two pages
    pages = [
        "| Column1 | Column2 |\n|---------|---------|\n| Value1  | Value2  |",
        "| Column3 | Column4 |\n|---------|---------|\n| Value3  | Value4  |",
    ]
    expected = [
        "| Column1 | Column2 |\n|---------|---------|\n| Value1  | Value2  |",
        "| Column3 | Column4 |\n|---------|---------|\n| Value3  | Value4  |",
    ]
    assert merger.merge_markdown_documents(pages) == expected

    # Case 3. Two tables with same header in two pages
    pages = [
        "| Column1 | Column2 |\n|---------|---------|\n| Value1  | Value2  |",
        "| Column1 | Column2 |\n|---------|---------|\n| Value3  | Value4  |",
    ]
    expected = [
        "| Column1 | Column2 |\n|---------|---------|\n| Value1  | Value2  |\n| Value3  | Value4  |"
    ]
    assert merger.merge_markdown_documents(pages) == expected

    # Case 4. Two tables in a mixed way in three pages
    pages = [
        "| Column1 | Column2 |\n|---------|---------|\n| Value1  | Value2  |",
        "| Value3  | Value4  |\n| Column3 | Column4 |\n|---------|---------|",
        "| Value5  | Value6  |",
    ]
    expected = [
        "| Column1 | Column2 |\n|---------|---------|\n| Value1  | Value2  |",
        "| Value3  | Value4  |\n| Column3 | Column4 |\n|---------|---------|\n| Value5  | Value6  |",
    ]
    assert merger.merge_markdown_documents(pages) == expected


def test_merge_code_blocks() -> None:
    merger = MarkdownMerger()
    pages = [
        "```python\nprint('Hello')",
        "print('World')\n```",
    ]
    expected = ["```python\nprint('Hello')\nprint('World')\n```"]
    assert merger.merge_markdown_documents(pages) == expected


def test_merge_sections_and_paragraphs() -> None:
    merger = MarkdownMerger()
    pages = [
        "## Header\n\nParagraph 1.\nParagraph 2.\n Paragraph",
        " 3.\n\nAnother paragraph.\n ## Header\n\nParagraph 1.",
        "# Header\n\nParagraph 1.",
    ]
    expected = [
        "## Header\n\nParagraph 1.\nParagraph 2.\n Paragraph 3.",
        "\nAnother paragraph.\n ## Header\n\nParagraph 1.",
        "# Header\n\nParagraph 1.",
    ]
    assert merger.merge_markdown_documents(pages) == expected


def test_mixed_content() -> None:
    merger = MarkdownMerger()
    pages = [
        "## Header\n\n- Item 1\n- Item 2",
        "- Item 3\n\n| Column1 | Column2 |\n|---------|---------|",
        "| Value1  | Value2  |\n\n```python\nprint('Hello')",
        "print('World')\n```\n\nAnother paragraph.",
    ]
    expected = [
        "## Header\n\n- Item 1\n- Item 2\n- Item 3",
        "\n| Column1 | Column2 |\n|---------|---------|\n| Value1  | Value2  |",
        "\n```python\nprint('Hello')\nprint('World')\n```",
        "\nAnother paragraph.",
    ]
    assert merger.merge_markdown_documents(pages) == expected


def test_empty_input() -> None:
    merger = MarkdownMerger()
    assert merger.merge_markdown_documents([]) == []


def test_single_page() -> None:
    merger = MarkdownMerger()
    pages = ["## Header\n\nContent"]
    assert merger.merge_markdown_documents(pages) == ["## Header\n\nContent"]


def test_make_continuation_transform() -> None:
    transformer = MakeContinuationMarkdownTransform.from_defaults()

    nodes = [
        TextNode(text="## Header\n\n- Item 1\n- Item 2"),
        TextNode(text="- Item 3\n\n| Column1 | Column2 |\n|---------|---------|"),
        TextNode(text="| Value1  | Value2  |\n\n```python\nprint('Hello')"),
        TextNode(text="print('World')\n```\n\nAnother paragraph."),
    ]
    expected = [
        "## Header\n\n- Item 1\n- Item 2\n- Item 3",
        "\n| Column1 | Column2 |\n|---------|---------|\n| Value1  | Value2  |",
        "\n```python\nprint('Hello')\nprint('World')\n```",
        "\nAnother paragraph.",
    ]

    transformed_nodes = transformer(nodes)
    transformed_texts = [node.get_content() for node in transformed_nodes]

    assert transformed_texts == expected
