from typing import TYPE_CHECKING

from llama_index.core.schema import TextNode

from private_gpt.components.ingest.transformations.remove_header_and_footer_transform import (
    RemoveHeaderAndFooterTransform,
)

if TYPE_CHECKING:
    from llama_index.core.schema import BaseNode


def test_remove_header_footer_without_keep_transform() -> None:
    transform = RemoveHeaderAndFooterTransform(
        keep_initial_header=False, keep_latest_footer=False
    )

    # Test case 1: Simple document with headers and footers
    test_content1 = """University of Example
Computer Science Department
Page 1

Main content here
More content
Even more content

Page 1 of 10"""

    test_content2 = """University of Example
Computer Science Department
Page 2

Different content
More details
Another paragraph

Page 2 of 10"""

    nodes = [TextNode(text=test_content1), TextNode(text=test_content2)]

    processed_nodes = transform(nodes)
    content1 = processed_nodes[0].get_content()
    content2 = processed_nodes[1].get_content()

    # Verify headers and footers are removed
    assert "University of Example" not in content1
    assert "Page 1 of 10" not in content1
    assert "University of Example" not in content2
    assert "Page 2 of 10" not in content2

    # Verify main content remains
    assert "Main content here" in content1
    assert "Different content" in content2


def test_remove_header_footer_transform() -> None:
    transform = RemoveHeaderAndFooterTransform(
        keep_initial_header=True, keep_latest_footer=True
    )

    # Test case 1: Simple document with headers and footers
    test_content1 = """University of Example
Computer Science Department
Page 1

Main content here
More content
Even more content

Page 1 of 10"""

    test_content2 = """University of Example
Computer Science Department
Page 2

Different content
More details
Another paragraph

Page 2 of 10"""

    nodes = [TextNode(text=test_content1), TextNode(text=test_content2)]

    processed_nodes = transform(nodes)
    content1 = processed_nodes[0].get_content()
    content2 = processed_nodes[1].get_content()

    # Verify headers and footers are removed
    assert "University of Example" in content1
    assert "Page 1 of 10" not in content1
    assert "University of Example" not in content2
    assert "Page 2 of 10" in content2

    # Verify main content remains
    assert "Main content here" in content1
    assert "Different content" in content2


def test_compare_method() -> None:
    transform = RemoveHeaderAndFooterTransform()

    # Test exact match
    assert transform._compare("Hello", "Hello") == 1.0

    # Test completely different strings
    assert transform._compare("Hello", "World") < 0.5

    # Test different strings with similar pattern
    result = transform._compare("Page 1", "Page 2!")
    assert 0.5 < result < 1.0

    result = transform._compare("Page 1", "Page 2")
    assert result == 1.0


def test_compute_logarithmic_weights() -> None:
    transform = RemoveHeaderAndFooterTransform()

    # Test with various input sizes
    weights1 = transform.compute_logarithmic_weights(3)
    weights2 = transform.compute_logarithmic_weights(5)

    assert len(weights1) == 3
    assert len(weights2) == 5
    assert max(weights1) <= 1.0
    assert min(weights1) > 0.0


def test_empty_input() -> None:
    nodes: list[BaseNode] = []
    transform = RemoveHeaderAndFooterTransform()
    result = transform(nodes)
    assert result == []


def test_pages_with_no_content() -> None:
    nodes = [TextNode(text=""), TextNode(text=""), TextNode(text="")]
    transform = RemoveHeaderAndFooterTransform()
    result = transform(nodes)
    assert all(node.text == "" for node in result)


def test_remove_headers() -> None:
    nodes = [
        TextNode(text="Header Line\nContent Line 1"),
        TextNode(text="Header Line\nContent Line 2"),
        TextNode(text="Header Line\nContent Line 3"),
    ]
    transform = RemoveHeaderAndFooterTransform(
        candidate_window=2, remove_footer=False, keep_initial_header=False
    )

    result = transform(nodes)
    expected_contents = ["Content Line 1", "Content Line 2", "Content Line 3"]
    for node, expected in zip(result, expected_contents, strict=False):
        assert node.text == expected


def test_remove_footers() -> None:
    nodes = [
        TextNode(text="Content Line 1\nFooter Line"),
        TextNode(text="Content Line 2\nFooter Line"),
        TextNode(text="Content Line 3\nFooter Line"),
    ]
    transform = RemoveHeaderAndFooterTransform(
        candidate_window=2, remove_header=False, keep_latest_footer=False
    )

    result = transform(nodes)
    expected_contents = ["Content Line 1", "Content Line 2", "Content Line 3"]
    for node, expected in zip(result, expected_contents, strict=False):
        assert node.text == expected


def test_no_headers_or_footers() -> None:
    """Test pages with no headers or footers."""
    nodes = [
        TextNode(text="Content Line 1"),
        TextNode(text="Content Line 2"),
        TextNode(text="Content Line 3"),
    ]
    transform = RemoveHeaderAndFooterTransform(
        candidate_window=2, keep_initial_header=False, keep_latest_footer=False
    )

    result = transform(nodes)
    expected_contents = ["Content Line 1", "Content Line 2", "Content Line 3"]
    for node, expected in zip(result, expected_contents, strict=False):
        assert node.text == expected


def test_empty_pages() -> None:
    """Test handling of empty pages."""
    nodes = [TextNode(text=""), TextNode(text="Content Line 2"), TextNode(text="")]
    transform = RemoveHeaderAndFooterTransform(candidate_window=2)

    result = transform(nodes)
    expected_contents = ["", "Content Line 2", ""]
    for node, expected in zip(result, expected_contents, strict=False):
        assert node.text == expected
