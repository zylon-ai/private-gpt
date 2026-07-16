import pytest

from private_gpt.components.ingest.transformations.markdown_normalization_transform import (
    MarkdownNormalizer,
)


def test_normalize_flat_list() -> None:
    """Test normalizing a flat Markdown list."""
    normalizer = MarkdownNormalizer(target_indent=2)
    content = "- Item 1\n  - Item 2\n    - Item 3"
    expected = "- Item 1\n  - Item 2\n    - Item 3"
    assert normalizer.normalize_markdown(content) == expected


def test_normalize_nested_list() -> None:
    """Test normalizing a nested Markdown list."""
    normalizer = MarkdownNormalizer(target_indent=2)
    content = "- Item 1\n  - Item 2\n      - Item 3\n        - Item 4"
    expected = "- Item 1\n  - Item 2\n    - Item 3\n      - Item 4"
    assert normalizer.normalize_markdown(content) == expected


def test_normalize_table() -> None:
    """Test that tables are not altered by normalization."""
    normalizer = MarkdownNormalizer(target_indent=2)
    content = "| Column1 | Column2 |\n|---------|---------|\n| Value1  | Value2  |"
    assert normalizer.normalize_markdown(content) == content


def test_normalize_code_block() -> None:
    """Test that code blocks are not altered by normalization."""
    normalizer = MarkdownNormalizer(target_indent=2)
    content = "```python\nprint('Hello')\nprint('World')\n```"
    assert normalizer.normalize_markdown(content) == content


def test_normalize_mixed_content() -> None:
    """Test normalizing mixed Markdown content."""
    normalizer = MarkdownNormalizer(target_indent=2)
    content = (
        "## Header\n\n- Item 1\n  - Item 2\n    - Item 3\n\n| Column1 | Column2 |\n|---------|---------|"
        "\n| Value1  | Value2  |\n\n```python\n print('Hello')\n    print('World')\n```"
    )
    expected = (
        "## Header\n\n- Item 1\n  - Item 2\n    - Item 3\n\n| Column1 | Column2 |\n|---------|---------|"
        "\n| Value1  | Value2  |\n\n```python\n print('Hello')\n    print('World')\n```"
    )
    assert normalizer.normalize_markdown(content) == expected


def test_normalize_bold_line_starts() -> None:
    """Test normalizing markdown content with lines starting with bold formatting."""
    normalizer = MarkdownNormalizer(target_indent=2)

    content = """**Chart Type**: Line chart

**Title**: Impact of When You Start Investing

**Axes**: X-axis: Age (25-65), Y-axis: Portfolio Value ($K)
**Legend**: Jack (blue), Jill (green), Joey (red)

**Data**:
    **Jack**: (25, 0) → (65, 550) → (Total, 96)
    **Jill**: (35, 0) → (65, 250) → (Total, 72)
    **Joey**: (45, 0) → (65, 100) → (Total, 48)

**Notes**: Source: Internal Analytics, significant growth with earlier starts, total contributions shown"""

    expected = """**Chart Type**: Line chart

**Title**: Impact of When You Start Investing

**Axes**: X-axis: Age (25-65), Y-axis: Portfolio Value ($K)
**Legend**: Jack (blue), Jill (green), Joey (red)

**Data**:
    **Jack**: (25, 0) → (65, 550) → (Total, 96)
    **Jill**: (35, 0) → (65, 250) → (Total, 72)
    **Joey**: (45, 0) → (65, 100) → (Total, 48)

**Notes**: Source: Internal Analytics, significant growth with earlier starts, total contributions shown"""

    assert normalizer.normalize_markdown(content) == expected


def test_normalize_mixed_bold_and_list() -> None:
    """Test normalizing markdown with both bold headers and nested list items."""
    normalizer = MarkdownNormalizer(target_indent=2)

    content = """**Investment Strategies**:
- Conservative approach
  - Bonds and CDs
    - Government bonds
- Aggressive approach
  - Growth stocks
    - Tech stocks
**Risk Assessment**: Medium to high"""

    expected = """**Investment Strategies**:
- Conservative approach
  - Bonds and CDs
    - Government bonds
- Aggressive approach
  - Growth stocks
    - Tech stocks
**Risk Assessment**: Medium to high"""

    assert normalizer.normalize_markdown(content) == expected


def test_normalize_bold_with_inconsistent_indentation() -> None:
    """Test normalizing bold-prefixed lines with inconsistent indentation."""
    normalizer = MarkdownNormalizer(target_indent=2)

    content = """**Summary**:
**Jack**: Early investor
    **Jill**: Mid-career starter
        **Joey**: Late starter
**Conclusion**: Start early for maximum growth"""

    expected = """**Summary**:
**Jack**: Early investor
    **Jill**: Mid-career starter
        **Joey**: Late starter
**Conclusion**: Start early for maximum growth"""

    assert normalizer.normalize_markdown(content) == expected


def test_empty_content() -> None:
    """Test handling of empty content."""
    normalizer = MarkdownNormalizer(target_indent=2)
    assert normalizer.normalize_markdown("") == ""


def test_no_lists_or_indents() -> None:
    """Test content with no lists or indentation."""
    normalizer = MarkdownNormalizer(target_indent=2)
    content = "## Header\n\nThis is a paragraph.\n\nAnother paragraph."
    assert normalizer.normalize_markdown(content) == content


@pytest.mark.parametrize(
    ("input_markdown", "expected_output"),
    [
        (
            # Test Case 1
            """\
- First list item
    * Nested item
        + Double nested item
  1. Numbered list
            """,
            """\
- First list item
  * Nested item
    + Double nested item
  1. Numbered list
            """,
        ),
        (
            # Test Case 2
            """\
- List item 1
    * Nested item
- List item 2
    * Another nested item
Some text in between
- Another first-level list
            """,
            """\
- List item 1
  * Nested item
- List item 2
  * Another nested item
Some text in between
- Another first-level list
            """,
        ),
        (
            # Test Case 3
            """\
1. First list
    - Sublist
    - Another sublist
2. Second list item
    - Different sublist
        - Even deeper nesting
            """,
            """\
1. First list
  - Sublist
  - Another sublist
2. Second list item
  - Different sublist
    - Even deeper nesting
            """,
        ),
    ],
)
def test_markdown_normalizer(input_markdown: str, expected_output: str) -> None:
    """Test MarkdownNormalizer with various input cases."""
    normalizer = MarkdownNormalizer(target_indent=2)
    normalized_output = normalizer.normalize_markdown(input_markdown)
    assert normalized_output.strip() == expected_output.strip()


def test_normalize_underline_headers() -> None:
    """Test normalizing underline-style headers."""
    normalizer = MarkdownNormalizer(target_indent=2)
    content = "Title\n=====\n\nSection\n-------\nNormal text"
    expected = "# Title\n\n## Section\nNormal text"
    assert normalizer.normalize_markdown(content) == expected


def test_normalize_mixed_headers() -> None:
    """Test normalizing mixed header styles."""
    normalizer = MarkdownNormalizer(target_indent=2)
    content = (
        "# Existing Header\n\nTitle\n=====\n\n## Another existing\n\nSection\n-------"
    )
    expected = "# Existing Header\n\n# Title\n\n## Another existing\n\n## Section"
    assert normalizer.normalize_markdown(content) == expected


def test_preserve_code_blocks_with_underlines() -> None:
    """Test that underlines in code blocks are not converted to headers."""
    normalizer = MarkdownNormalizer(target_indent=2)
    content = "# Header\n\n```markdown\nTitle\n=====\n```\n\nNormal text"
    assert normalizer.normalize_markdown(content) == content


def test_preserve_table_with_underlines() -> None:
    """Test that table separators are not converted to headers."""
    normalizer = MarkdownNormalizer(target_indent=2)
    content = "| Header |\n|--------|\n| Value  |"
    assert normalizer.normalize_markdown(content) == content


def test_normalize_lists_with_headers() -> None:
    """Test normalizing lists combined with headers."""
    normalizer = MarkdownNormalizer(target_indent=2)
    content = "Title\n=====\n- Item 1\n  - Item 2\nSection\n-------\n    - Item 3"
    expected = "# Title\n- Item 1\n  - Item 2\n## Section\n    - Item 3"
    assert normalizer.normalize_markdown(content) == expected


def test_normalize_tabs_and_bold_headers() -> None:
    """Test normalizing content with tabs and bold headers."""
    normalizer = MarkdownNormalizer(target_indent=2)
    content = (
        "**Header 1**\n"
        "===========\n"
        "\n"
        "* List item 1\n"
        "\t* Nested with tab\n"
        "\t\t* Double nested\n"
        "\n"
        "**Header 2**\n"
        "-----------\n"
    )
    expected = (
        "# **Header 1**\n"
        "\n"
        "* List item 1\n"
        "  * Nested with tab\n"
        "    * Double nested\n"
        "\n"
        "## **Header 2**"
    )
    assert normalizer.normalize_markdown(content) == expected
