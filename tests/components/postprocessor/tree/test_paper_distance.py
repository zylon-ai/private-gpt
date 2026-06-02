import textwrap

import pytest

from private_gpt.components.ingest.metadata_helper import MetadataNode
from private_gpt.components.ingest.transformations.markdown_to_tree_transform import (
    MarkdownTreeNodeParser,
)
from private_gpt.components.postprocessor.tree_expansion.paper_distance import (
    PaperDistanceAlg,
)
from private_gpt.components.readers.nodes.document_node import DocumentRootNode
from private_gpt.components.readers.nodes.table_node import TableNode
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode

_hit = "HIT"  # The node where the expansion will begin
_fail_marker = "FAIL"  # Special node that should not be found in the result

_basic = """\
HIT
"""

_generic = f"""\
# Title 1
## Section 1
Sample 1
Sample 2
Sample 3
Sample 4
## Section 2
Sample 10
Sample 11
{_hit}
## Section 3
Sample 12
"""

_subsection_under_hit = f"""\
# Title 1
## Section 1
Some text
{_hit}
Some text
### Hit subsection 2
Some subsection 1

Some subsection 2

### Hit subsection 2.1
Some subsection 1
Some subsection 2

### Hit subsection 3
Some subsection 3
Some subsection 4
Some subsection 5

### Hit subsection 4
Some subsection 3
Some subsection 4

## {_fail_marker}
{_fail_marker}
Some text
Some text
"""

_many_sections = f"""\
# Title 1
# Title 2
# Title 3
# Title 4
# Title 5
# Title 6
# {_fail_marker}
Should not reach

# Title Before
Some text before
Some text before
Some text before
Some text before

## Level 2
Deep text 2.0
Deep text 2.0

### Level 3
Deep text 3
Deep text 3
Deep text 3

## Level 2.1
Deep text 2.1
{_fail_marker}

# Title of hit
{_hit}

# Title After

Some text after
Some text after
Some text after
Some text after

## Subtitle after

Some text after

# {_fail_marker}
Should not reach

# Title 12
# Title 13
# Title 14
# Title 15
# Title 16
"""

_table = f"""\
The above-listed deferred tax assets, recognised in the accompanying consolidated statement of financial position, reflect the Parent's directors' belief, underpinned by its best estimate of the Group's earnings, that it is probable that future taxable profit will be available against which the assets can be utilised.

The reconciliation of deferred taxes at the beginning and end of 2023 and 2022:

{_hit}

2023

| Item | Deferred tax assets | Deferred tax liabilities |
| - | - | - |
| Balance at 28 February 2022 | 188,899 | 846,363 |
| Temporary differences: | | |
| Current fiscal year | 187,615 | 3,129 |
| Previous fiscal years | (134,404) | (11,391) |
| Total at 29 February 2024 | 242,110 | 838,101 |
| In € 000 | | |

2022

| Item | Deferred tax assets | Deferred tax liabilities |
| - | - | - |
| Balance at 28 February 2022 | 267,645 | 828,623 |
| Temporary differences: | | |
| Current fiscal year | 29,364 | 25,085 |
| Previous fiscal years | (108,110) | (7,345) |
| Total at 28 February 2023 | 188,899 | 846,363 |

In € 000

The balance of deferred tax assets arising from temporary differences in each company are provided in Appendix II.

The unused tax credits capitalised and the corresponding expiration dates:

| Type of credit | 2023 | Year of expiry | 2022 | Year of expiry |
| - | - | - | - | - |
| Intra-group double taxation credit | 130,300 | Unlimited | 129,665 | Unlimited |
| International double taxation credit | 1,029 | Unlimited | 448 | Unlimited |
| Investment tax credit | 2,628 | 2035-2036 | 75,755 | 2025-2039 |
| Donations to NGOs | - | - | 2,275 | 2022-2031 |
| Re-investment tax credit | - | - | 27,510 | 2022-2030 |
| Reversal of temporary measures | - | - | 11,023 | Unlimited |
| Balance at year-end | 133,957 | | 246,676 | |

In € 000
"""

_all_samples = [_basic, _generic, _subsection_under_hit, _many_sections]


def debug_print(tree: TreeNode, result: [str]):
    flat = tree.flatten()
    flat_output = ""
    for node in flat:
        flat_output += (
            node.get_content(TreeMetadataMode.RAG) if node.id_ in result else ""
        )

    print("FINAL CONTEXT")
    print("====================")
    print(flat_output)
    print("====================")


def _parse_tree(sample: str) -> (TreeNode, TreeNode):
    parser = MarkdownTreeNodeParser(include_metadata=True)
    tree = parser.parse(textwrap.dedent(sample))
    # find the hit node
    flat = list(tree.flatten())
    # Give a fake token count
    for node in flat:
        node.metadata[MetadataNode.TOKEN_COUNT] = len(node.get_content().split())
        node.excluded_llm_metadata_keys = [MetadataNode.TOKEN_COUNT]
        # Special case for tables and root, they report 0 tokens
        if node.isinstance(TableNode) or node.isinstance(DocumentRootNode):
            node.metadata[MetadataNode.TOKEN_COUNT] = 0
    hit_node = None
    for node in reversed(flat):
        if _hit in node.get_content():
            hit_node = node
            break
    assert hit_node is not None, "Hit node not found"
    return tree, hit_node


def _common_assertions(tree: TreeNode, hit: TreeNode, token_limit: int, result: [str]):
    flat = list(tree.flatten())
    selected = [node for node in flat if node.id_ in result]
    token_count = sum([node.token_count for node in selected])
    assert token_count <= token_limit, "Token count exceeds limit"
    assert hit.id_ in result, "Hit node not in result"


def _assert_no_fail_nodes_in_result(tree: TreeNode, result: [str]):
    flat = reversed(list(tree.flatten()))
    fail_nodes = [
        node for node in flat if _fail_marker in node.get_content(TreeMetadataMode.RAG)
    ]
    assert not any(
        node.id_ in result for node in fail_nodes
    ), "Fail node found in result"


@pytest.mark.parametrize("markdown", _all_samples)
def test_paper_distance_basic_checks(markdown: str):
    tree, hit = _parse_tree(markdown)
    alg = PaperDistanceAlg()
    token_limit = 50
    result = alg.fill_window(hit, token_limit)
    _common_assertions(tree, hit, token_limit, result)


@pytest.mark.parametrize("markdown", _all_samples)
def test_paper_distance_includes_full_tree_if_there_is_room(markdown: str):
    tree, hit = _parse_tree(markdown)
    alg = PaperDistanceAlg()
    result = alg.fill_window(hit, 1000000)
    _common_assertions(tree, hit, 1000000, result)
    assert len(result) == len(list(tree.flatten()))


def test_paper_distance_favors_subsections():
    tree, hit = _parse_tree(_subsection_under_hit)
    alg = PaperDistanceAlg()
    result = alg.fill_window(hit, 50)
    _common_assertions(tree, hit, 50, result)
    _assert_no_fail_nodes_in_result(tree, result)


def test_paper_distance_with_no_tokens_left_returns_empty_list():
    _, hit = _parse_tree(_generic)
    alg = PaperDistanceAlg()
    result = alg.fill_window(hit, 0)
    assert len(result) == 0


def test_paper_distance_expands_close_sections():
    tree, hit = _parse_tree(_many_sections)
    alg = PaperDistanceAlg()
    token_limit = 500
    result = alg.fill_window(hit, token_limit)
    debug_print(tree, result)
    _common_assertions(tree, hit, token_limit, result)
    _assert_no_fail_nodes_in_result(tree, result)


def test_paper_distance_expands_tables():
    tree, hit = _parse_tree(_table)
    alg = PaperDistanceAlg()
    token_limit = 550
    result = alg.fill_window(hit, token_limit)
    debug_print(tree, result)
    _common_assertions(tree, hit, token_limit, result)
    _assert_no_fail_nodes_in_result(tree, result)
