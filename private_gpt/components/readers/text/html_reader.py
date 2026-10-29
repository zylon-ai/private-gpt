import re

from bs4 import BeautifulSoup
from markdownify import markdownify  # ty:ignore[unresolved-import]

from private_gpt.components.markdown.markdown_helper import MarkdownHelper
from private_gpt.components.readers.text.text_reader import TextReader


class HtmlReader(TextReader):
    def _clean_up_html(self, html_content: str) -> str:
        """Convert HTML content to clean text, handling complex documents."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove common non-content elements
        for element in soup(["script", "style", "meta", "link", "title", "head"]):
            element.decompose()

        # Remove XBRL/XML namespace elements and hidden content
        for element in soup.find_all(
            ["ix:header", "ix:hidden", "ix:references", "ix:resources"]
        ):
            element.decompose()

        # Remove elements with display:none or hidden attributes
        for element in soup.find_all(style=re.compile(r"display\s*:\s*none")):
            element.decompose()

        for element in soup.find_all(
            attrs={"style": re.compile(r"display\s*:\s*none")}
        ):
            element.decompose()

        # Unwrap the remaining namespaced tags (`ix:nonFraction`, Word's `st1:City`):
        # their markup is not HTML, but their text is the document's own. Inline
        # XBRL keeps every reported figure in such a tag, so removing it removed
        # the numbers.
        for element in soup.find_all(re.compile(r"^[^:]*:")):
            element.unwrap()

        # Get text content from body if it exists, otherwise from entire document
        body = soup.find("body")
        content_element = body if body else soup

        return str(content_element)

    def _convert_html_to_markdown(self, html_content: str) -> str:
        """Convert HTML content to markdown."""
        markdown: str = markdownify(  # type: ignore[no-untyped-call]
            html_content, heading_style="ATX", bullets="-", strip=["script", "style"]
        ).strip()
        markdown = MarkdownHelper.sanitize_markdown(markdown)
        return markdown

    def _process_content(self, content: str) -> str:
        content = self._clean_up_html(content)
        content = self._convert_html_to_markdown(content)
        return content
