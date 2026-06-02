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
        markdown = "1. First item\n" "2. Second item"
        self._round_trip_test(markdown)

        markdown = "- First item\n" "- Second item"
        self._round_trip_test(markdown, exact=False)

        markdown = "+ First item\n" "\t+ Second item"
        self._round_trip_test(markdown, exact=False)

    def test_code_blocks(self) -> None:
        """Test inline, indented, and fenced code blocks."""
        markdown = "Inline `code`"
        self._round_trip_test(markdown)

        markdown = "```\n" "Sample text\n" "```"
        self._round_trip_test(markdown)

        markdown = "```js \n" "var foo = 'bar';\n" "```"
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
        markdown = "| 0 | 1 |\n" "| - | - |\n" "| Alice | 25 |\n" "| Bob | 30 |"
        self._round_trip_test(markdown)

        # Case 4: Has default header with header row
        markdown = (
            "| 0 | 1 |\n"
            "| - | - |\n"
            "| Name | Age |\n"
            "| Alice | 25 |\n"
            "| Bob | 30 |"
        )
        root = self.parser.parse(markdown)
        serialized = self._serialize_tree(root)
        expected = (
            "| Name | Age |\n" "| - | - |\n" "| Alice | 25 |\n" "| Bob | 30 |\n\n"
        )
        assert serialized.strip() == expected.strip()

        # Case 5. Empty headers
        markdown = "| | |\n" "| - | - |\n" "| Alice | 25 |\n" "| Bob | 30 |"
        root = self.parser.parse(markdown)
        serialized = self._serialize_tree(root)
        expected = (
            "| Unknown_1 | Unknown_2 |\n"
            "| - | - |\n"
            "| Alice | 25 |\n"
            "| Bob | 30 |\n\n"
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
        markdown = (
            "> Blockquote level 1\n" "> Blockquote level 2\n" "> Blockquote level 3"
        )
        self._round_trip_test(markdown)
        markdown = (
            "> Blockquote level 1\n" ">> Blockquote level 2\n" ">>> Blockquote level 3"
        )
        self._round_trip_test(markdown)

    def test_typographic_replacements(self) -> None:
        """Test typographic replacements."""
        markdown = "(c) (C) (r) (R) (tm) (TM)\n" "Smart quotes: 'single' and \"double\""
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
        assert (
            expected.strip() == new_markdown.strip()
        ), "Sanitized headings should remove excessive spaces and normalize formatting."

    def test_sanity_paragraph_format(self) -> None:
        """Test the sanitization of paragraphs with inconsistent line breaks."""
        markdown = "Paragraph one.\n\n   Paragraph two with extra indentation.\n\n"
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "Paragraph one.\n\nParagraph two with extra indentation.\n\n"
        assert (
            expected.strip() == new_markdown.strip()
        ), "Sanitized paragraphs should have consistent formatting and spacing."

    def test_sanity_image_format(self) -> None:
        """Test the sanitization of image syntax."""
        markdown = "![  Alt Text  ]( http://example.com )"
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "![Alt Text](http://example.com)"
        assert (
            expected.strip() == new_markdown.strip()
        ), "Sanitized images should have no unnecessary spaces and correct syntax."

    def test_sanity_link_format(self) -> None:
        """Test the sanitization of link syntax."""
        markdown = "[  Link Text  ]( http://example.com )"
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "[Link Text](http://example.com)\n\n"
        assert (
            expected.strip() == new_markdown.strip()
        ), "Sanitized links should remove extra spaces and preserve valid syntax."

    def test_sanity_bold_format(self) -> None:
        """Test the sanitization of bold syntax."""
        markdown = "**Bold Text** and __More Bold__"
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "Bold Text and More Bold\n\n"
        assert (
            expected.strip() == new_markdown.strip()
        ), "Sanitized bold should remove asterisks/underscores and preserve text content."

    def test_sanity_italic_format(self) -> None:
        """Test the sanitization of italic syntax."""
        markdown = "*Italic Text* and _More Italic_"
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "Italic Text and More Italic\n\n"
        assert (
            expected.strip() == new_markdown.strip()
        ), "Sanitized italic should remove single asterisks/underscores and preserve text content."

    def test_sanity_bold_italic_format(self) -> None:
        """Test the sanitization of combined bold and italic syntax."""
        markdown = "***Bold Italic*** and ___More Bold Italic___"
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "Bold Italic and More Bold Italic\n\n"
        assert (
            expected.strip() == new_markdown.strip()
        ), "Sanitized bold-italic should remove triple asterisks/underscores and preserve text content."

    def test_sanity_mixed_format(self) -> None:
        """Test the sanitization of mixed markdown formatting."""
        markdown = "**Bold** with *italic* and ***bold-italic*** mixed"
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "Bold with italic and bold-italic mixed\n\n"
        assert (
            expected.strip() == new_markdown.strip()
        ), "Sanitized mixed formatting should remove all markdown syntax and preserve text content."

    def test_sanity_item_format(self) -> None:
        markdown = "* ● Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily."
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "* Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily.\n"
        assert (
            expected == new_markdown
        ), "Sanitized items should remove extra icons and preserve valid syntax."

        markdown = "* ◦ Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily."
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "* Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily.\n"
        assert (
            expected == new_markdown
        ), "Sanitized items should remove extra spaces and preserve valid syntax."

        markdown = "*     ◦ Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily."
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "* Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily.\n"
        assert (
            expected == new_markdown
        ), "Sanitized items should remove extra spaces and preserve valid syntax."

        markdown = "*     + Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily."
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "* Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily.\n"
        assert (
            expected == new_markdown
        ), "Sanitized items should remove extra spaces and preserve valid syntax."

        markdown = "* - Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily."
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "* Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily.\n"
        assert (
            expected == new_markdown
        ), "Sanitized items should remove extra spaces and preserve valid syntax."

        markdown = "  -     1. Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily."
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "1. Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily.\n"
        assert (
            expected == new_markdown
        ), "Sanitized items should remove extra spaces and preserve valid syntax."

        markdown = "  5.     + Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily."
        new_markdown = self._round_trip_test(markdown, check=False)
        expected = "5. Hypertension: Diagnosed at age 32, currently managed with losartan 50 mg once daily.\n"
        assert (
            expected == new_markdown
        ), "Sanitized items should remove extra spaces and preserve valid syntax."

    def test_sample_markdown(self) -> None:
        """Test the parser on the sample markdown."""
        self._round_trip_test(SAMPLE_MARKDOWN, exact=False)
