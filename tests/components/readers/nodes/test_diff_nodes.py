import pandas as pd

from private_gpt.components.readers.nodes.diff_node import DiffNode
from private_gpt.components.readers.nodes.table_node import TableNode
from private_gpt.components.readers.nodes.text_node import TextNode
from private_gpt.components.readers.nodes.tree_node import TreeNode


def run_diff_test(
    ref_node: TreeNode,
    other_node: TreeNode,
    expected_output: str | None,
) -> None:
    """Helper function to run a diff test."""
    result = DiffNode.from_nodes(
        ref_node,
        other_node,
    )
    content = result.get_content().strip() if result else None
    expected_output = expected_output.strip() if expected_output else None
    assert content == expected_output, f"Expected:\n{expected_output}\n\nGot:\n{result}"


def test_identical_texts() -> None:
    """Test identical texts should return None."""
    ref_node = TextNode(text="This is a simple line of text.")
    other_node = TextNode(text="This is a simple line of text.")
    run_diff_test(ref_node, other_node, None)


def test_identical_tables() -> None:
    """Test identical markdown tables."""
    ref_node = TableNode(
        df=pd.DataFrame({"Header 1": ["Cell 1"], "Header 2": ["Cell 2"]})
    )
    other_node = TableNode(
        df=pd.DataFrame({"Header 1": ["Cell 1"], "Header 2": ["Cell 2"]})
    )
    run_diff_test(ref_node, other_node, None)


def test_table_with_extra_row() -> None:
    """Test a table with an extra row in the reference text."""
    ref_df = pd.DataFrame(
        {"Header 1": ["Cell 1", "Cell 3"], "Header 2": ["Cell 2", "Cell 4"]}
    )
    other_df = pd.DataFrame({"Header 1": ["Cell 3"], "Header 2": ["Cell 4"]})
    ref_node = TableNode(df=ref_df)
    other_node = TableNode(df=other_df)
    expected_output = (
        "| Header 1 | Header 2 |\n"
        "| - | - |\n"
        "@@ Skipped 1 lines @@\n"
        "| Cell 3 | Cell 4 |\n\n"
    )
    run_diff_test(ref_node, other_node, expected_output)


def test_with_mixture_content() -> None:
    ref_df = pd.DataFrame(
        {
            "Header 1": ["Cell 1", "Cell 3", "Cell 3", "Cell 3", "Cell 5", "Cell 3"],
            "Header 2": ["Cell 2", "Cell 4", "Cell 4", "Cell 4", "Cell 6", "Cell 4"],
        }
    )
    other_df = pd.DataFrame(
        {
            "Header 1": ["Cell 1", "Cell 3", "Cell 3", "Cell 3"],
            "Header 2": ["Cell 2", "Cell 4", "Cell 4", "Cell 4"],
        }
    )

    ref_node = TableNode(df=ref_df)
    other_node = TableNode(df=other_df)
    expected_output = (
        "| Header 1 | Header 2 |\n"
        "| - | - |\n"
        "| Cell 1 | Cell 2 |\n"
        "| Cell 3 | Cell 4 |\n"
        "| Cell 3 | Cell 4 |\n"
        "| Cell 3 | Cell 4 |\n"
        "@@ Skipped 2 lines @@\n"
    )
    run_diff_test(ref_node, other_node, expected_output)


def test_empty_texts() -> None:
    """Test both texts being empty."""
    ref_node = TextNode(text="")
    other_node = TextNode(text="")
    run_diff_test(ref_node, other_node, None)
