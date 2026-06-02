import textwrap

import pytest

from private_gpt.components.ingest.metadata_helper import MetadataNode
from private_gpt.components.ingest.transformations.markdown_to_tree_transform import (
    MarkdownTreeNodeParser,
)
from private_gpt.components.postprocessor.tree_expansion.document_expander import (
    DocumentTreeExpander,
)
from private_gpt.components.readers.nodes.document_node import DocumentRootNode
from private_gpt.components.readers.nodes.table_node import TableNode
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode

# Markers
_hit = "HIT"  # The node where the expansion will begin
_fail_marker = "FAIL"  # Special node that should not be found in the result

# Existing Markdown Samples
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

## Level 2.1
Deep text 2.1

Deep text 2.2

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

_sibling_rollback_right = f"""\
# Title 1
## Section 1
Content 1
## Section 2
{_hit}
## Section 3
Small content
## {_fail_marker}
Large content
Large content
Large content
## Section 4
Small content
## {_fail_marker}
Large content
Large content
"""

_sibling_rollback_left = f"""\
# Title 1
## {_fail_marker}
Large content
Large content
## Section 2
Small content
## Section 3
{_hit}
## {_fail_marker}
Large content
Large content
Large content
Large content
Large content
Large content
Large content
Large content
Large content
## Section 4
Small content
"""

_parent_sibling_rollback = f"""\
# Title 1
## Section A
Content A
### Subsection A.1
{_hit}
### Subsection A.2
Content A.2
## {_fail_marker}
Large content
Large content
Large content
## Section B
Small content B
### Subsection B.1
Content B.1
"""

_expansion_downward_blocked = f"""\
# {_hit}
## Section 1
Content 1
## Section 2
Large content that should
not be included due to the tokens
and presence of fail marker
But other content should still be
included if within token limits
## {_fail_marker}
Content 2
"""

_expansion_right_blocked = f"""\
# Section 1
## {_hit}
Content 1
## {_fail_marker}
Large content that should
not be included due to the tokens
and presence of fail marker
But other content should still be
included if within token limits
## Section 2
Content 2
"""

_expansion_upwards_left_blocked = f"""\
# Title 1
## {_fail_marker}
Large content that should
not be included due to the tokens
## Section 2
Small content 2
## Current Section
{_hit}
## Section 3
Small content 3
"""

_expansion_upwards_right_blocked = f"""\
# Title 1
## Section 1
Small content 1
## Current Section
{_hit}
## Section 3
Small content 3
## Section 4
Small content 4
## {_fail_marker}
Large content that should
not be included due to the tokens
"""

_expansion_horizontal_blocked = f"""\
# Title 1
## {_fail_marker}
Block left
## Dummy section
Small content
## Dummy section
Small content
## Section 2
{_hit}
## Dummy section
Small content
## Dummy section
Small content
## {_fail_marker}
Block right
"""


_all_samples = [
    _basic,
    _generic,
    _subsection_under_hit,
    _many_sections,
    _table,
    _sibling_rollback_right,
    _sibling_rollback_left,
    _parent_sibling_rollback,
]

_all_expansion_samples = [
    _expansion_downward_blocked,
    _expansion_upwards_left_blocked,
    _expansion_upwards_right_blocked,
    _expansion_horizontal_blocked,
]


def debug_print(tree: TreeNode, result: [str]) -> None:
    flat = tree.flatten()
    flat_output = ""
    for node in flat:
        flat_output += (
            node.get_content(TreeMetadataMode.RAG) if node.id_ in result else ""
        )

    print("====================")
    print("FINAL CONTEXT")
    print("====================")
    print(flat_output)
    print("====================")


def _parse_tree(sample: str) -> (TreeNode, TreeNode):
    parser = MarkdownTreeNodeParser(include_metadata=True)
    tree = parser.parse(textwrap.dedent(sample))
    # Find the HIT node
    flat = list(tree.flatten())
    # Assign fake token counts
    for node in flat:
        node.metadata[MetadataNode.TOKEN_COUNT] = len(
            node.get_content(TreeMetadataMode.RAG).split()
        )
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


def _common_assertions(
    tree: TreeNode, hit: TreeNode, token_limit: int, result: tuple[set[str], int]
) -> None:
    flat = list(tree.flatten())
    node_ids, token_count = result
    selected = [node for node in flat if node.id_ in node_ids]
    sum_token_count = sum([node.token_count for node in selected])
    assert sum_token_count <= token_count, "Token count does not match"
    assert token_count <= token_limit, "Token count exceeds limit"
    assert hit.id_ in node_ids, "Hit node not in result"


def _assert_no_fail_nodes_in_result(tree: TreeNode, result: [str]) -> None:
    flat = reversed(list(tree.flatten()))
    fail_nodes = [
        node for node in flat if _fail_marker in node.get_content(TreeMetadataMode.RAG)
    ]
    assert not any(
        node.id_ in result for node in fail_nodes
    ), "Fail node found in result"


def create_fake_document_expander(
    node: TreeNode, max_tokens: int
) -> DocumentTreeExpander:
    return DocumentTreeExpander(node, max_tokens)


# ============================
# Original Test Cases
# ============================


@pytest.mark.parametrize("markdown", _all_samples)
def test_document_expander_basic_checks(markdown: str) -> None:
    tree, hit = _parse_tree(markdown)
    token_limit = 50
    alg = create_fake_document_expander(hit, token_limit)
    result = alg.fill_window()
    _common_assertions(tree, hit, token_limit, result)


@pytest.mark.parametrize("markdown", _all_samples)
def test_document_expander_includes_full_tree_if_there_is_room(markdown: str) -> None:
    tree, hit = _parse_tree(markdown)
    token_limit = 1000000
    alg = create_fake_document_expander(hit, token_limit)
    result = alg.fill_window()
    _common_assertions(tree, hit, token_limit, result)
    assert len(result[0]) == len(list(tree.flatten()))


def test_document_expander_favors_subsections() -> None:
    tree, hit = _parse_tree(_subsection_under_hit)
    token_limit = 50
    alg = create_fake_document_expander(hit, token_limit)
    result = alg.fill_window()
    _common_assertions(tree, hit, token_limit, result)
    _assert_no_fail_nodes_in_result(tree, result)


def test_document_expander_with_no_tokens_left_returns_empty_list() -> None:
    _, hit = _parse_tree(_generic)
    token_limit = 0
    alg = create_fake_document_expander(hit, token_limit)
    result, _ = alg.fill_window()
    assert len(result) == 0


def test_document_expander_expands_close_sections() -> None:
    tree, hit = _parse_tree(_many_sections)
    token_limit = 20
    alg = create_fake_document_expander(hit, token_limit)
    result = alg.fill_window()
    debug_print(tree, result)
    _common_assertions(tree, hit, token_limit, result)
    _assert_no_fail_nodes_in_result(tree, result)


def test_document_expander_expands_tables() -> None:
    tree, hit = _parse_tree(_table)
    token_limit = 550
    alg = create_fake_document_expander(hit, token_limit)
    result = alg.fill_window()
    debug_print(tree, result)
    _common_assertions(tree, hit, token_limit, result)
    _assert_no_fail_nodes_in_result(tree, result)


@pytest.mark.parametrize("markdown", _all_expansion_samples)
def test_expansion(markdown: str) -> None:
    tree, hit = _parse_tree(markdown)
    token_limit = 12
    alg = create_fake_document_expander(hit, token_limit)
    result = alg.fill_window()
    debug_print(tree, result)
    _common_assertions(tree, hit, token_limit, result)
    _assert_no_fail_nodes_in_result(tree, result)
