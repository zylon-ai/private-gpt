from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import NavigableString, Tag
from llama_index.core.schema import MetadataMode, TransformComponent
from mistune import HTMLRenderer, create_markdown  # ty:ignore[unresolved-import]
from pydantic import Field

from private_gpt.components.ingest.processors.df_preprocessor import (
    DataFramePreprocessor,
)
from private_gpt.components.markdown.markdown_helper import MarkdownHelper
from private_gpt.components.readers.nodes import TextNode
from private_gpt.components.readers.nodes.document_node import DocumentRootNode
from private_gpt.components.readers.nodes.image_node import ImageNode
from private_gpt.components.readers.nodes.list_node import ListItemNode, ListNode
from private_gpt.components.readers.nodes.section_node import SectionNode
from private_gpt.components.readers.nodes.table_node import TableNode, TableRowNode
from private_gpt.components.readers.nodes.v2.document_node_v2 import DocumentRootNodeV2

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from llama_index.core.schema import BaseNode

    from private_gpt.components.readers.nodes.tree_node import TreeNode


class MarkdownTreeNodeParser(TransformComponent):
    include_metadata: bool = Field(
        default=True,
        description="Whether to include metadata in the nodes.",
    )

    markdown_reader: Any = Field(
        default=create_markdown(
            renderer=HTMLRenderer(), plugins=["strikethrough", "table"]
        ),
        description="Markdown parser.",
        exclude=True,
    )
    root: DocumentRootNode = Field(
        default=DocumentRootNode(),
        description="Root node.",
        exclude=True,
    )
    section_stack: list[SectionNode] = Field(
        default_factory=list,
        description="Stack of section nodes.",
        exclude=True,
    )

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        new_nodes = []
        for node in nodes:
            markdown = node.get_content(MetadataMode.NONE)
            parsed_node = self.parse(markdown, node.metadata)
            new_nodes.append(parsed_node)
        return new_nodes

    def _find_parent_for_level(self, level: int) -> TreeNode:
        while (
            self.section_stack
            and self.section_stack[-1].metadata.get("header_level", 0) >= level
        ):
            self.section_stack.pop()
        return self.section_stack[-1] if self.section_stack else self.root

    def _get_current_parent(self) -> TreeNode:
        return self.section_stack[-1] if self.section_stack else self.root

    def _parse_table(self, table_html: str) -> pd.DataFrame:
        """Parse one-line HTML table into pandas DataFrame using regex."""
        # Extract headers with one regex match
        headers = [
            text.strip()
            for text in re.findall(r"<th[^>]*>(.*?)</th>", table_html, re.IGNORECASE)
        ]

        # Extract all cells in one regex match and chunk them into rows
        cells = [
            cell.strip()
            for cell in re.findall(r"<td[^>]*>(.*?)</td>", table_html, re.IGNORECASE)
        ]

        # Calculate number of columns from headers
        num_cols = len(headers)

        # Convert flat list of cells into rows
        rows = [cells[i : i + num_cols] for i in range(0, len(cells), num_cols)]

        process_table = DataFramePreprocessor(
            try_cast_to_numeric=True,
            try_cast_to_datetime=True,
        )
        return process_table.preprocess_table_data(rows, headers)

    def _generate_table_node(self, df: pd.DataFrame) -> TreeNode:
        table_node = TableNode(
            df=df,
        )

        return table_node

    def _generate_table_row_node(self, df: pd.DataFrame) -> Iterator[TreeNode]:
        header: list[str] = df.columns.tolist()
        for _, row in df.iterrows():
            content: list[Any] = [
                "" if pd.isna(value) else value for value in row.tolist()
            ]
            node = TableRowNode(
                header=[str(h) for h in header],
                content=content,
            )
            yield node

    def _process_list(self, element: Tag, parent: TreeNode) -> None:
        """Process list elements (ul/ol) and their items."""
        list_type = "ordered" if element.name == "ol" else "unordered"
        list_node = ListNode(
            extra_info={"list_type": list_type},
            excluded_llm_metadata_keys=["list_type"],
            excluded_embed_metadata_keys=["list_type"],
            content_type="text/markdown",
        )
        parent.add_child(list_node)
        items = element.find_all("li", recursive=False)
        start_number = int(element.get("start", "1"))  # type: ignore

        for index, item in enumerate(items):
            self._process_list_item(item, list_node, list_type, start_number + index)

    def _process_list_item(
        self,
        item: Tag,
        parent: ListNode,
        list_type: str,
        item_number: int,
        indent_level: int = 0,
    ) -> None:
        """Process a single list item, including nested lists."""
        prefix = f"{item_number}. " if list_type == "ordered" else "* "
        prefix = ("  " * indent_level) + prefix
        item_content = self._extract_item_content(item)
        markdown_line = MarkdownHelper.sanitize_markdown(f"{prefix}{item_content}\n")

        item_node = ListItemNode(
            text=markdown_line,
            extra_info={
                "list_type": list_type,
                "item_number": item_number if list_type == "ordered" else None,
                "indent_level": indent_level,
            },
            excluded_llm_metadata_keys=["list_type", "item_number", "indent_level"],
            excluded_embed_metadata_keys=["list_type", "item_number", "indent_level"],
            content_type="text/markdown",
        )
        parent.add_child(item_node)

        # Process any nested lists within this list item
        nested_lists = item.find_all(["ul", "ol"], recursive=False)
        for nested_list in nested_lists:
            nested_list_type = "ordered" if nested_list.name == "ol" else "unordered"
            nested_start_number = int(nested_list.get("start", "1"))
            nested_items = nested_list.find_all("li", recursive=False)
            nested_list_node = ListNode(
                extra_info={"list_type": nested_list_type},
                excluded_llm_metadata_keys=["list_type"],
                excluded_embed_metadata_keys=["list_type"],
                content_type="text/markdown",
            )
            item_node.add_child(nested_list_node)

            for idx, nested_item in enumerate(nested_items):
                self._process_list_item(
                    nested_item,
                    nested_list_node,
                    nested_list_type,
                    nested_start_number + idx,
                    indent_level=indent_level + 1,
                )

    def _process_blockquote(self, element: Tag, current_level: int = 1) -> TextNode:
        current_node = TextNode(
            text="",
            extra_info={"type": "blockquote", "level": current_level},
            excluded_llm_metadata_keys=["type", "level"],
            excluded_embed_metadata_keys=["type", "level"],
            content_type="text/markdown",
        )

        text_lines = []
        for child in element.find_all(["blockquote", "p"], recursive=False):
            if child.name == "p":
                lines = child.get_text(strip=True).split("\n")
                text_lines.extend([f"{'>' * current_level} {line}\n" for line in lines])
            elif child.name == "blockquote":
                nested_node = self._process_blockquote(child, current_level + 1)
                current_node.add_child(nested_node)

        current_node.text = MarkdownHelper.sanitize_markdown("\n".join(text_lines))
        return current_node

    def _extract_item_content(self, item: Tag) -> str:
        """Extract content from a list item."""
        content_parts = []
        for child in item.contents:
            if isinstance(child, NavigableString):
                content_parts.append(child.strip())
            elif isinstance(child, Tag):
                if child.name in ["ul", "ol"]:
                    continue
                content_parts.append(self._convert_tag_to_markdown(child))
        return " ".join(content_parts).strip()

    def _add_node_to_hierarchy(self, node: TreeNode) -> None:
        """Add node to the appropriate parent in the hierarchy."""
        if isinstance(node, SectionNode):
            header_level = node.metadata.get("header_level", 1)
            parent = self._find_parent_for_level(header_level)
            parent.add_child(node)
            self.section_stack.append(node)
        else:
            self._get_current_parent().add_child(node)

    def _convert_tag_to_markdown(self, element: Tag | NavigableString) -> str:
        """Convert a BeautifulSoup element to markdown while preserving formatting."""
        from markdownify import markdownify as md  # ty:ignore[unresolved-import]

        if isinstance(element, NavigableString):
            return element
        else:
            markdown: str = md(str(element), heading_style="ATX")
            markdown = markdown.replace("\\", "")
            markdown = MarkdownHelper.sanitize_markdown(markdown)
            return markdown

    def _process_element(self, element: Tag | NavigableString) -> None:
        """Process a single HTML element and create appropriate nodes."""
        node: TreeNode

        if isinstance(element, NavigableString):
            text = element.strip()
            if text:
                node = TextNode(
                    text=text,
                    content_type="text/markdown",
                )
                self._add_node_to_hierarchy(node)
            return

        if element.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
            level = int(element.name[1])
            header_text = element.get_text(strip=False)
            markdown = f'{"#" * level} {header_text}\n\n'
            node = SectionNode(
                text=MarkdownHelper.sanitize_markdown(markdown),
                extra_info={"header_level": level},
                excluded_llm_metadata_keys=["header_level"],
                excluded_embed_metadata_keys=["header_level"],
            )
            self._add_node_to_hierarchy(node)
        elif element.name == "table":
            df = self._parse_table(str(element))
            table_node = self._generate_table_node(df)
            self._add_node_to_hierarchy(table_node)
            table_node.add_children(*(self._generate_table_row_node(df)))
        elif element.name in ["ul", "ol"]:
            self._process_list(element, self._get_current_parent())
        elif element.name == "pre":
            code_element = element.find("code")
            if code_element:
                language = ""
                for cls in code_element.get("class", []):  # type: ignore
                    if cls.startswith("language-"):
                        language = cls.split("language-")[1]
                        break
                code_text = code_element.get_text(strip=True)
                language = language.strip()
                node = TextNode(
                    text=MarkdownHelper.sanitize_markdown(
                        f"\n```{language}\n{code_text}\n```\n"
                    ),
                    extra_info={"type": "code_block", "language": language},
                    excluded_llm_metadata_keys=["type", "language"],
                    content_type="text/markdown",
                )
            else:
                code_text = element.get_text(strip=True)
                node = TextNode(
                    text=MarkdownHelper.sanitize_markdown(f"\n```\n{code_text}\n```\n"),
                    extra_info={"type": "code_block"},
                    excluded_llm_metadata_keys=["type"],
                    excluded_embed_metadata_keys=["type"],
                    content_type="text/markdown",
                )
            self._add_node_to_hierarchy(node)
        elif element.name == "code" and element.parent.name != "pre":  # type: ignore
            inline_code = element.get_text(strip=True)
            node = TextNode(
                text=MarkdownHelper.sanitize_markdown(f"`{inline_code}`"),
                extra_info={"type": "inline_code"},
                excluded_llm_metadata_keys=["type"],
                excluded_embed_metadata_keys=["type"],
                content_type="text/markdown",
            )
            self._add_node_to_hierarchy(node)
        elif element.name == "code":
            if element.parent.name != "pre":  # type: ignore
                code_text = element.get_text(strip=False)
                node = TextNode(
                    text=MarkdownHelper.sanitize_markdown(f"`{code_text}`"),
                    extra_info={"type": "inline_code"},
                    excluded_llm_metadata_keys=["type"],
                    excluded_embed_metadata_keys=["type"],
                    content_type="text/markdown",
                )
                self._add_node_to_hierarchy(node)
        elif element.name == "blockquote":
            node = self._process_blockquote(element)
            self._add_node_to_hierarchy(node)
        elif element.name == "img":
            alt_text: str = str(element.get("alt", ""))
            src: str = str(element.get("src", ""))
            node = ImageNode(
                alt_text=alt_text,
                image=src,
                extra_info={"type": "image", "alt_text": alt_text, "src": src},
                excluded_llm_metadata_keys=["type", "alt_text", "src"],
                excluded_embed_metadata_keys=["type", "alt_text", "src"],
                content_type="text/markdown",
            )
            self._add_node_to_hierarchy(node)
        elif element.name == "p":
            text = self._convert_tag_to_markdown(element)
            if text:
                node = TextNode(
                    text=text,
                    content_type="text/markdown",
                )
                self._add_node_to_hierarchy(node)
        else:
            for child in element.contents:
                self._process_element(child)  # type: ignore

    def _process_html(self, html: str) -> None:
        """Process HTML and build node hierarchy."""
        # Use lxml for parsing HTML
        # It's faster and more lenient than the default parser
        # https://thehftguy.com/2020/07/28/making-beautifulsoup-parsing-10-times-faster/

        soup = BeautifulSoup(html, "lxml")

        # Process each element in the HTML
        for element in soup.contents:
            self._process_element(element)  # type: ignore

    def _update_headers_metadata(self, node: TreeNode, headers: list[str]) -> None:
        """Update headers metadata for each node."""
        if isinstance(node, SectionNode):
            headers = [*headers, node.text.strip("# ").strip()]
        node.metadata["headers"] = headers
        node.excluded_llm_metadata_keys.append("headers")
        node.excluded_embed_metadata_keys.append("headers")
        for child in node.children or []:
            self._update_headers_metadata(child, headers)

    def _copy_metadata_to_children(self, node: TreeNode) -> None:
        """Copy metadata from parent to children."""
        for child in node.children or []:
            child.metadata.update(node.metadata)
            child.excluded_llm_metadata_keys = list(
                {
                    *node.excluded_llm_metadata_keys,
                    *child.excluded_llm_metadata_keys,
                }
            )
            child.excluded_embed_metadata_keys = list(
                {
                    *node.excluded_embed_metadata_keys,
                    *child.excluded_embed_metadata_keys,
                }
            )
            self._copy_metadata_to_children(child)

    def parse(
        self, markdown: str, extra_info: dict[str, Any] | None = None
    ) -> DocumentRootNode:
        """Parse Markdown into a DocumentRoot tree."""
        self.root = DocumentRootNodeV2()
        self.section_stack = []

        if self.include_metadata and extra_info:
            self.root.metadata = extra_info
            self.root.excluded_llm_metadata_keys = list(extra_info.keys())
            self.root.excluded_embed_metadata_keys = list(extra_info.keys())

        # Parse the markdown into HTML and process it
        html = self.markdown_reader(MarkdownHelper.sanitize_markdown(markdown))
        self._process_html(html)
        self._update_headers_metadata(self.root, [])

        # Copy metadata to children
        if self.include_metadata:
            self._copy_metadata_to_children(self.root)

        return self.root

    @classmethod
    def from_defaults(
        cls,
        include_metadata: bool = True,
    ) -> MarkdownTreeNodeParser:
        return cls(
            include_metadata=include_metadata,
        )
