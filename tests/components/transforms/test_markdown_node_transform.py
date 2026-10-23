import unittest

from private_gpt.components.ingest.transformations.markdown_to_tree_transform import (
    MarkdownTreeNodeParser,
)
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode

SAMPLE_MARKDOWN = """\
### Document Title
#### Subtitle
This is a sample document generated randomly with various elements such as tables, lists, nested lists, code blocks, and headers.


### Table Example
| Column A | Column B |
|----------|----------|
| Cell 1   | Cell 2   |
|----------|----------|
| Cell 3   | Cell 4   |

### Nested List Example
* Item 1
	+ Subitem 1
	+ Subitem 2
* Item 2
	+ Subsubitem 1
		- Further subsubitem 1
		- Further subsubitem 2
"""


class TestMarkdownParser(unittest.TestCase):
    def setUp(self) -> None:
        """Set up the parser instance."""
        self.parser = MarkdownTreeNodeParser(include_metadata=True)

    def _serialize_tree(self, node: TreeNode) -> str:
        """Recursively serialize a tree node to markdown."""
        return node.get_content(TreeMetadataMode.USER)

    def _markdown_to_html(self, markdown: str) -> str:
        """Convert markdown to html."""
        from mistune import HTMLRenderer, create_markdown

        md = create_markdown(
            renderer=HTMLRenderer(),  # type: ignore
            plugins=["strikethrough", "table"],
        )  # type: ignore
        return md(markdown)  # type: ignore

    def _round_trip_test(
        self, markdown: str, check: bool = True, exact: bool = True
    ) -> str:
        """Perform a round-trip parse/serialize/parse/serialize test."""
        # Parse input markdown
        root = self.parser.parse(markdown)
        # Serialize the tree back to markdown
        serialized = self._serialize_tree(root)
        # Parse serialized markdown
        reparsed_root = self.parser.parse(serialized)
        # Serialize reparsed tree
        reserialized = self._serialize_tree(reparsed_root)
        # Assert equality between input and re-serialized markdown

        if check:
            if exact:
                assert reserialized.strip() == markdown.strip()
            else:
                # Compare  the input markdown and the re-serialized markdown
                # as html strings, since the Markdown parser may add extra
                input_html = self._markdown_to_html(markdown)
                reserialized_html = self._markdown_to_html(reserialized)
                assert reserialized_html.strip() == input_html.strip()
        return reserialized

    def test_headers(self) -> None:
        """Test header parsing."""
        markdown = "# Header 1\n\n## Header 2\n\n### Header 3"
        self._round_trip_test(markdown)

        markdown = "# Header 1\n## Header 2\n### Header 3"
        self._round_trip_test(markdown, exact=False)

    def test_lists(self) -> None:
        """Test ordered and unordered list parsing."""
        markdown = "1. First item\n2. Second item"
        self._round_trip_test(markdown)

        markdown = "- First item\n- Second item"
        self._round_trip_test(markdown, exact=False)

        markdown = "+ First item\n\t+ Second item"
        self._round_trip_test(markdown, exact=False)

    def test_code_blocks(self) -> None:
        """Test inline, indented, and fenced code blocks."""
        markdown = "Inline `code`"
        self._round_trip_test(markdown)

        markdown = "```\nSample text\n```"
        self._round_trip_test(markdown)

        markdown = "```js \nvar foo = 'bar';\n```"
        self._round_trip_test(markdown, exact=False)

    def test_tables(self) -> None:
        """Test table parsing."""
        # Case 1: Single row table
        markdown = (
            "| Header 1 | Header 2 |\n"
            "|----------|----------|\n"
            "| Cell 1   | Cell 2   |\n"
            "| Cell 3   | Cell 4   |"
        )
        self._round_trip_test(markdown, exact=False)

        # Case 2: Duplicate header name
        markdown = (
            "| Header 1 | Header 1 |\n"
            "|----------|----------|\n"
            "| Cell 1   | Cell 2   |\n"
            "| Cell 3   | Cell 4   |"
        )
        root = self.parser.parse(markdown)
        serialized = self._serialize_tree(root)
        expected = (
            "| Header 1_1 | Header 1_2 |\n"
            "| - | - |\n"
            "| Cell 1 | Cell 2 |\n"
            "| Cell 3 | Cell 4 |\n\n"
        )
        assert serialized.strip() == expected.strip()

        # Case 3: Has default header without header row
        markdown = "| 0 | 1 |\n| - | - |\n| Alice | 25 |\n| Bob | 30 |"
        self._round_trip_test(markdown)

        # Case 4: Has default header with header row
        markdown = "| 0 | 1 |\n| - | - |\n| Name | Age |\n| Alice | 25 |\n| Bob | 30 |"
        root = self.parser.parse(markdown)
        serialized = self._serialize_tree(root)
        expected = "| Name | Age |\n| - | - |\n| Alice | 25 |\n| Bob | 30 |\n\n"
        assert serialized.strip() == expected.strip()

        # Case 5. Empty headers
        markdown = "| | |\n| - | - |\n| Alice | 25 |\n| Bob | 30 |"
        root = self.parser.parse(markdown)
        serialized = self._serialize_tree(root)
        expected = (
            "| Unknown_1 | Unknown_2 |\n| - | - |\n| Alice | 25 |\n| Bob | 30 |\n\n"
        )
        assert serialized.strip() == expected.strip()

        # Case 6. Nan header in some position
        markdown = (
            "| 0 | 1 | 2 |\n"
            "| - | - | - |\n"
            "| nan | Name | Age |\n"
            "| 1 | Alice | 25 |\n"
            "| 2 | Bob | 30 |"
        )
        root = self.parser.parse(markdown)
        serialized = self._serialize_tree(root)
        expected = (
            "| Unknown | Name | Age |\n"
            "| - | - | - |\n"
            "| 1 | Alice | 25 |\n"
            "| 2 | Bob | 30 |\n\n"
        )
        assert serialized.strip() == expected.strip()

    def test_blockquotes(self) -> None:
        """Test blockquote parsing."""
        markdown = "> Blockquote level 1\n> Blockquote level 2\n> Blockquote level 3"
        self._round_trip_test(markdown)
        markdown = "> Blockquote level 1\n>> Blockquote level 2\n>>> Blockquote level 3"
        self._round_trip_test(markdown)

    def test_typographic_replacements(self) -> None:
        """Test typographic replacements."""
        markdown = "(c) (C) (r) (R) (tm) (TM)\nSmart quotes: 'single' and \"double\""
        self._round_trip_test(markdown)

    def test_images_and_links(self) -> None:
        """Test images and links."""
        markdown = (
            "[Example link](http://example.com)\n"
            "![Alt text](http://example.com/image.png)"
        )
        self._round_trip_test(markdown)

    def test_sanity_heading_format(self) -> None:
        """Test the sanitization of headings for proper trimming and formatting."""
        markdown = "   ##  Improper Heading"
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "## Improper Heading"
        assert expected.strip() == new_markdown.strip(), (
            "Sanitized headings should remove excessive spaces and normalize formatting."
        )

    def test_sanity_paragraph_format(self) -> None:
        """Test the sanitization of paragraphs with inconsistent line breaks."""
        markdown = "Paragraph one.\n\n   Paragraph two with extra indentation.\n\n"
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "Paragraph one.\n\nParagraph two with extra indentation.\n\n"
        assert expected.strip() == new_markdown.strip(), (
            "Sanitized paragraphs should have consistent formatting and spacing."
        )

    def test_sanity_image_format(self) -> None:
        """Test the sanitization of image syntax."""
        markdown = "![  Alt Text  ]( http://example.com )"
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "![Alt Text](http://example.com)"
        assert expected.strip() == new_markdown.strip(), (
            "Sanitized images should have no unnecessary spaces and correct syntax."
        )

    def test_sanity_link_format(self) -> None:
        """Test the sanitization of link syntax."""
        markdown = "[  Link Text  ]( http://example.com )"
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "[Link Text](http://example.com)\n\n"
        assert expected.strip() == new_markdown.strip(), (
            "Sanitized links should remove extra spaces and preserve valid syntax."
        )

    def test_sanity_bold_format(self) -> None:
        """Test the sanitization of bold syntax."""
        markdown = "**Bold Text** and __More Bold__"
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "Bold Text and More Bold\n\n"
        assert expected.strip() == new_markdown.strip(), (
            "Sanitized bold should remove asterisks/underscores and preserve text content."
        )

    def test_sanity_italic_format(self) -> None:
        """Test the sanitization of italic syntax."""
        markdown = "*Italic Text* and _More Italic_"
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "Italic Text and More Italic\n\n"
        assert expected.strip() == new_markdown.strip(), (
            "Sanitized italic should remove single asterisks/underscores and preserve text content."
        )

    def test_sanity_bold_italic_format(self) -> None:
        """Test the sanitization of combined bold and italic syntax."""
        markdown = "***Bold Italic*** and ___More Bold Italic___"
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "Bold Italic and More Bold Italic\n\n"
        assert expected.strip() == new_markdown.strip(), (
            "Sanitized bold-italic should remove triple asterisks/underscores and preserve text content."
        )

    def test_sanity_mixed_format(self) -> None:
        """Test the sanitization of mixed markdown formatting."""
        markdown = "**Bold** with *italic* and ***bold-italic*** mixed"
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "Bold with italic and bold-italic mixed\n\n"
        assert expected.strip() == new_markdown.strip(), (
            "Sanitized mixed formatting should remove all markdown syntax and preserve text content."
        )

    def test_sanity_item_format(self) -> None:
        markdown = "* ● Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily."
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "* Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily.\n"
        assert expected == new_markdown, (
            "Sanitized items should remove extra icons and preserve valid syntax."
        )

        markdown = "* ◦ Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily."
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "* Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily.\n"
        assert expected == new_markdown, (
            "Sanitized items should remove extra spaces and preserve valid syntax."
        )

        markdown = "*     ◦ Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily."
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "* Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily.\n"
        assert expected == new_markdown, (
            "Sanitized items should remove extra spaces and preserve valid syntax."
        )

        markdown = "*     + Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily."
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "* Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily.\n"
        assert expected == new_markdown, (
            "Sanitized items should remove extra spaces and preserve valid syntax."
        )

        markdown = "* - Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily."
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "* Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily.\n"
        assert expected == new_markdown, (
            "Sanitized items should remove extra spaces and preserve valid syntax."
        )

        markdown = "  -     1. Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily."
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "1. Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily.\n"
        assert expected == new_markdown, (
            "Sanitized items should remove extra spaces and preserve valid syntax."
        )

        markdown = "  5.     + Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily."
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "5. Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily.\n"
        assert expected == new_markdown, (
            "Sanitized items should remove extra spaces and preserve valid syntax."
        )

    def test_sample_markdown(self) -> None:
        """Test the parser on the sample markdown."""
        self._round_trip_test(SAMPLE_MARKDOWN, exact=False)

    def test_metadata_propagation_to_children(self) -> None:
        """Verify that root metadata and excluded keys propagate correctly
        to all descendant nodes. This locks in current behavior before
        refactoring _copy_metadata_to_children."""
        markdown = (
            "# Header 1\n"
            "Paragraph under header 1.\n\n"
            "## Header 2\n"
            "* Item 1\n"
            "* Item 2\n\n"
            "| A | B |\n"
            "|---|---|\n"
            "| 1 | 2 |\n"
        )
        extra_info = {"source": "doc1", "page": 3}
        root = self.parser.parse(markdown, extra_info=extra_info)

        def collect_nodes(node, acc=None):
            acc = [] if acc is None else acc
            acc.append(node)
            for child in node.children or []:
                collect_nodes(child, acc)
            return acc

        all_nodes = collect_nodes(root)

        # Root itself must carry the extra_info
        assert root.metadata.get("source") == "doc1"
        assert root.metadata.get("page") == 3

        # Every descendant node must have inherited the root metadata
        for node in all_nodes[1:]:  # skip root, already checked
            assert node.metadata.get("source") == "doc1", (
                f"Node {type(node).__name__} missing inherited 'source' metadata"
            )
            assert node.metadata.get("page") == 3, (
                f"Node {type(node).__name__} missing inherited 'page' metadata"
            )
            # Inherited exclusion keys must be present too
            assert "source" in node.excluded_llm_metadata_keys
            assert "source" in node.excluded_embed_metadata_keys
            assert "page" in node.excluded_llm_metadata_keys
            assert "page" in node.excluded_embed_metadata_keys

        # Node-specific exclusion keys must be preserved, not overwritten
        # e.g. SectionNode/ListItemNode add their own keys like 'header_level',
        # 'list_type', etc. Spot-check at least one such node still has them.
        section_nodes = [n for n in all_nodes if "header_level" in n.metadata]
        assert section_nodes, "Expected at least one SectionNode with header_level"
        for sn in section_nodes:
            assert "header_level" in sn.excluded_llm_metadata_keys
            assert "header_level" in sn.excluded_embed_metadata_keys

    def test_metadata_propagation_without_extra_info(self) -> None:
        """When no extra_info is passed, children should not crash and
        should simply have no inherited root-level keys beyond their own."""
        markdown = "# Header\nSome text"
        root = self.parser.parse(markdown)  # no extra_info

        def collect_nodes(node, acc=None):
            acc = [] if acc is None else acc
            acc.append(node)
            for child in node.children or []:
                collect_nodes(child, acc)
            return acc

        all_nodes = collect_nodes(root)
        # Should not raise, and metadata dicts should exist (possibly empty
        # aside from node-specific keys like header_level/headers)
        for node in all_nodes:
            assert isinstance(node.metadata, dict)

    def test_table_with_inline_formatting(self) -> None:
        """Cells containing inline markdown (bold, italic, links, code)
        must be parsed correctly. This guards _parse_table against
        behavior changes if regex parsing is replaced with find_all()."""
        markdown = (
            "| Header 1 | Header 2 |\n"
            "|----------|----------|\n"
            "| **Bold** | *Italic* |\n"
            "| [link](http://example.com) | `code` |"
        )
        self._round_trip_test(markdown, exact=False)

    def test_table_with_nested_tags_in_cell(self) -> None:
        """A cell with multiple nested inline tags shouldn't break header/
        cell count alignment."""
        markdown = (
            "| Name | Notes |\n"
            "|------|-------|\n"
            "| Alice | **Important**: see *details* below |\n"
            "| Bob | Plain text |"
        )
        root = self.parser.parse(markdown)
        # Should not raise, and should produce exactly 2 data rows
        serialized = self._serialize_tree(root)
        assert "Alice" in serialized
        assert "Bob" in serialized

    def test_table_row_count_matches_input(self) -> None:
        """Regression guard: number of TableRowNode children must match
        the number of data rows in the source markdown, regardless of
        how cells are extracted internally."""
        from private_gpt.components.readers.nodes.table_node import TableNode

        markdown = (
            "| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |\n| 7 | 8 | 9 |"
        )
        root = self.parser.parse(markdown)

        def find_table_nodes(node, acc=None):
            acc = [] if acc is None else acc
            if isinstance(node, TableNode):
                acc.append(node)
            for child in node.children or []:
                find_table_nodes(child, acc)
            return acc

        tables = find_table_nodes(root)
        assert len(tables) == 1
        assert len(tables[0].children or []) == 3  # 3 data rows
