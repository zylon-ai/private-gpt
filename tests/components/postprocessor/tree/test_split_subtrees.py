import pandas as pd
import pytest

from private_gpt.components.postprocessor.tree_expansion.split_subtrees import (
    SplitSubtreeAlg,
)
from private_gpt.components.readers.nodes import (
    DocumentRootNode,
    ListNode,
    SectionNode,
    TableNode,
    TextNode,
)


def create_multilevel_sections() -> DocumentRootNode:
    """Creates a tree with multiple nested section levels."""
    root = DocumentRootNode()

    # First main section with subsections
    section1 = SectionNode(text="Main Section 1")
    root.add_child(section1)
    text1 = TextNode(text="Main section 1 content")
    section1.add_child(text1)

    # Subsections under section1
    subsection1_1 = SectionNode(text="Subsection 1.1")
    section1.add_child(subsection1_1)
    text1_1 = TextNode(text="Subsection 1.1 content")
    subsection1_1.add_child(text1_1)

    subsection1_2 = SectionNode(text="Subsection 1.2")
    section1.add_child(subsection1_2)
    text1_2 = TextNode(text="Subsection 1.2 content")
    subsection1_2.add_child(text1_2)

    # Second main section with deeper nesting
    section2 = SectionNode(text="Main Section 2")
    root.add_child(section2)

    subsection2_1 = SectionNode(text="Subsection 2.1")
    section2.add_child(subsection2_1)

    # Even deeper nesting
    subsubsection2_1_1 = SectionNode(text="SubSubsection 2.1.1")
    subsection2_1.add_child(subsubsection2_1_1)
    text2_1_1 = TextNode(text="Deep nested content")
    subsubsection2_1_1.add_child(text2_1_1)

    # Update references
    root.update_references()

    return root


def create_no_sections() -> DocumentRootNode:
    """Creates a tree with no sections, just different types of content."""
    root = DocumentRootNode()

    # Add some text nodes
    text1 = TextNode(text="First paragraph")
    root.add_child(text1)

    text2 = TextNode(text="Second paragraph")
    root.add_child(text2)

    # Add a list
    list_node = ListNode()
    root.add_child(list_node)
    list_item1 = TextNode(text="List item 1")
    list_item2 = TextNode(text="List item 2")
    list_node.add_child(list_item1)
    list_node.add_child(list_item2)

    # Add a table
    df = pd.DataFrame({"Column1": ["A", "B"], "Column2": [1, 2]})
    table = TableNode(df=df, description="Sample table")
    root.add_child(table)

    # Update references
    root.update_references()

    return root


def create_multiple_sections_same_level() -> DocumentRootNode:
    """Creates a tree with multiple sections at the same level."""
    root = DocumentRootNode()

    # Section 1 with content
    section1 = SectionNode(text="Section 1")
    root.add_child(section1)
    text1 = TextNode(text="Content for section 1")
    section1.add_child(text1)

    # Section 2 with list
    section2 = SectionNode(text="Section 2")
    root.add_child(section2)
    list_node = ListNode()
    section2.add_child(list_node)
    list_item = TextNode(text="List item in section 2")
    list_node.add_child(list_item)

    # Section 3 with table
    section3 = SectionNode(text="Section 3")
    root.add_child(section3)
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    table = TableNode(df=df, description="Table in section 3")
    section3.add_child(table)

    # Update references
    root.update_references()

    return root


def create_mixed_depth_sections() -> DocumentRootNode:
    """Creates a tree with sections at different depths mixed together."""
    root = DocumentRootNode()

    # First section with deep nesting
    section1 = SectionNode(text="Section 1")
    root.add_child(section1)

    subsection1_1 = SectionNode(text="Subsection 1.1")
    section1.add_child(subsection1_1)

    subsubsection1_1_1 = SectionNode(text="SubSubsection 1.1.1")
    subsection1_1.add_child(subsubsection1_1_1)
    text1 = TextNode(text="Deep nested content")
    subsubsection1_1_1.add_child(text1)

    # Second section at root level
    section2 = SectionNode(text="Section 2")
    root.add_child(section2)
    text2 = TextNode(text="Root level content")
    section2.add_child(text2)

    # Third section with medium nesting
    section3 = SectionNode(text="Section 3")
    root.add_child(section3)

    subsection3_1 = SectionNode(text="Subsection 3.1")
    section3.add_child(subsection3_1)
    text3 = TextNode(text="Medium nested content")
    subsection3_1.add_child(text3)

    # Update references
    root.update_references()

    return root


@pytest.mark.parametrize(
    ("tree_setup", "expected_min_subtrees", "description"),
    [
        (create_no_sections, 1, "No sections"),
        (create_multiple_sections_same_level, 3, "Multiple sections at same level"),
        (create_mixed_depth_sections, 3, "Mixed depth sections"),
        (create_multilevel_sections, 3, "Multilevel nested sections"),
    ],
)
def test_various_tree_structures(
    tree_setup,
    expected_min_subtrees: int,
    description: str,
) -> None:
    """Test the splitter with various tree structures."""
    alg = SplitSubtreeAlg()
    tree = tree_setup()
    tree.print_tree()  # For debugging
    subtrees = alg.split_subtree(tree)

    print(f"\nTesting: {description}")
    print(f"Generated {len(subtrees)} subtrees")
    for i, st in enumerate(subtrees):
        print(f"\nSubtree {i}:")
        st.print_tree()

    # Basic quantity check
    assert len(subtrees) >= expected_min_subtrees, (
        f"Expected at least {expected_min_subtrees} subtrees for {description}, "
        f"but got {len(subtrees)}"
    )

    # Verify each subtree is valid
    for subtree in subtrees:
        assert isinstance(subtree, DocumentRootNode), (
            "Subtree root should be DocumentRoot"
        )
        # Check that each subtree has at least one child
        assert len(subtree.children) > 0, "Subtree should not be empty"


def test_multilevel_section_hierarchy() -> None:
    """Detailed test for multilevel section hierarchy preservation."""
    alg = SplitSubtreeAlg()
    tree = create_multilevel_sections()
    tree.print_tree()  # For debugging
    subtrees = alg.split_subtree(tree)

    print("\nTesting: Multilevel section hierarchy")
    print(f"Generated {len(subtrees)} subtrees")
    for i, st in enumerate(subtrees):
        print(f"\nSubtree {i}:")
        st.print_tree()

    # Verify that deep nested sections stay together
    for subtree in subtrees:
        if len(subtree.children) > 0 and isinstance(subtree.children[0], SectionNode):
            section = subtree.children[0]
            # If this section has subsections, verify the parent-child relationship
            if any(isinstance(child, SectionNode) for child in section.children):
                # Verify parent references are correct
                for child in section.children:
                    if isinstance(child, SectionNode):
                        assert child.parent.id_ == section.id_, (
                            "Parent-child relationship broken"
                        )


def test_no_sections_integrity() -> None:
    """Detailed test for trees without sections."""
    alg = SplitSubtreeAlg()
    tree = create_no_sections()
    tree.print_tree()  # For debugging
    subtrees = alg.split_subtree(tree)

    print("\nTesting: No sections")
    print(f"Generated {len(subtrees)} subtrees")
    for i, st in enumerate(subtrees):
        print(f"\nSubtree {i}:")
        st.print_tree()

    # Should return exactly one subtree as there are no section splits
    assert len(subtrees) == 1, "Tree without sections should not be split"

    # Verify all content is preserved
    original_nodes = list(tree.flatten())
    result_nodes = list(subtrees[0].flatten())
    assert len(result_nodes) >= len(original_nodes), (
        "Content was lost in no-sections tree"
    )

    # Verify the types of nodes are preserved
    original_types = [type(n) for n in original_nodes]
    result_types = [type(n) for n in result_nodes]
    result_types = result_types[len(result_types) - len(original_types) :]

    assert original_types == result_types, "Node types changed in no-sections tree"


def test_multiple_sections_relationships() -> None:
    """Detailed test for multiple sections at the same level."""
    alg = SplitSubtreeAlg()
    tree = create_multiple_sections_same_level()
    tree.print_tree()  # For debugging
    subtrees = alg.split_subtree(tree)

    print("\nTesting: Multiple sections at same level")
    print(f"Generated {len(subtrees)} subtrees")
    for i, st in enumerate(subtrees):
        print(f"\nSubtree {i}:")
        st.print_tree()

    # Verify each section is in its own subtree
    section_counts = [
        sum(1 for child in subtree.flatten() if isinstance(child, SectionNode))
        for subtree in subtrees
    ]

    # Each subtree should have exactly one section
    assert all(count == 1 for count in section_counts), (
        "Each subtree should contain exactly one section"
    )

    # Verify content type preservation
    for subtree in subtrees:
        section = next(
            (child for child in subtree.children if isinstance(child, SectionNode)),
            None,
        )
        if section:
            # Verify that the section's content is preserved
            if "Section 2" in section.text:
                assert any(isinstance(node, ListNode) for node in section.flatten()), (
                    "List content not preserved in Section 2"
                )
            elif "Section 3" in section.text:
                assert any(isinstance(node, TableNode) for node in section.flatten()), (
                    "Table content not preserved in Section 3"
                )
