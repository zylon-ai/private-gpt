import asyncio
import logging
import re
from collections.abc import Callable, Sequence
from typing import Any, cast

from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent
from pydantic import Field

logger = logging.getLogger(__name__)


class AddTitleHeaderTransform(TransformComponent):
    """Adds h1 title to each document if it doesn't start with one."""

    def __init__(self, default_title: str = "Document", batch_size: int = 100):
        super().__init__()
        self._default_title = default_title
        self._batch_size = batch_size

        self._h1_pattern = re.compile(r"^#\s+", re.MULTILINE)
        self._first_line_pattern = re.compile(r"^(.+?)(?:\n|$)", re.MULTILINE)
        self._markdown_formatting_pattern = re.compile(r"[*_`~]")
        self._whitespace_pattern = re.compile(r"\s+")
        self._trailing_punctuation_pattern = re.compile(r"[.,:;!?]+$")

    @classmethod
    def from_defaults(
        cls, default_title: str = "Document", batch_size: int = 100
    ) -> "AddTitleHeaderTransform":
        return cls(default_title=default_title, batch_size=batch_size)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        return asyncio.run(self.acall(nodes, **kwargs))

    async def acall(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        if not nodes:
            return nodes

        processed_nodes: list[BaseNode] = []

        for i in range(0, len(nodes), self._batch_size):
            batch = nodes[i : i + self._batch_size]
            batch_tasks = [
                self._process_single_node(node_idx, node)
                for node_idx, node in enumerate(batch, start=i)
            ]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for node, result in zip(batch, batch_results, strict=False):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to add title header: {result}")
                    processed_nodes.append(node)
                else:
                    processed_nodes.append(cast(BaseNode, result))

        return processed_nodes

    async def _process_single_node(self, idx: int, node: BaseNode) -> BaseNode:
        try:
            content = node.get_content(MetadataMode.NONE)
            if not content or not content.strip():
                return node

            content = content.strip()
            title = f"{self._default_title} {idx + 1}"
            modified_content = f"# {title}\n\n{content.strip()}\n\n\n"

            processed_node = node.model_copy()
            processed_node.set_content(modified_content)

            return processed_node

        except Exception as e:
            logger.error(f"Error adding title header: {e}")
            return node

    def _clean_title(self, title: str) -> str:
        title = self._markdown_formatting_pattern.sub("", title)
        title = self._whitespace_pattern.sub(" ", title)
        title = self._trailing_punctuation_pattern.sub("", title)
        return title.strip()


class ReduceHeaderLevelsTransform(TransformComponent):
    """Reduces header levels by N (adds N more # symbols to each header).

    Example with reduce_by=1:
        # Header 1     ->  ## Header 1
        ## Header 2    ->  ### Header 2
        ### Header 3   ->  #### Header 3

    Headers that would exceed h6 are converted to bold text.
    """

    reduce_by: int = Field(
        default=1,
        description="Number of levels to increase header levels by. Must be non-negative.",
    )
    predicate: Callable[[str], bool] = Field(
        default=lambda content: True,
        description=(
            "A callable that takes the node content as input and returns True if the transform should be applied, False otherwise."
        ),
    )

    def __init__(
        self,
        reduce_by: int = 1,
        predicate: Callable[[str], bool] | None = None,
        batch_size: int = 100,
    ):
        super().__init__(  # type: ignore
            reduce_by=reduce_by,
            predicate=predicate or (lambda content: True),
        )
        self._batch_size = batch_size

        if reduce_by < 0:
            raise ValueError("reduce_by must be non-negative")

        # Matches any header level with capture groups for level and text
        self._header_pattern = re.compile(r"^(#{1,6})\s+(.+?)$", re.MULTILINE)

    @classmethod
    def from_defaults(
        cls,
        reduce_by: int = 1,
        predicate: Callable[[str], bool] | None = None,
        batch_size: int = 100,
    ) -> "ReduceHeaderLevelsTransform":
        return cls(reduce_by=reduce_by, predicate=predicate, batch_size=batch_size)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        return asyncio.run(self.acall(nodes, **kwargs))

    async def acall(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        if not nodes:
            return nodes

        processed_nodes: list[BaseNode] = []

        for i in range(0, len(nodes), self._batch_size):
            batch = nodes[i : i + self._batch_size]
            batch_tasks = [self._process_single_node(node) for node in batch]

            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for node, result in zip(batch, batch_results, strict=False):
                if isinstance(result, Exception):
                    logger.warning(f"Failed to reduce header levels: {result}")
                    processed_nodes.append(node)
                else:
                    processed_nodes.append(cast(BaseNode, result))

        return processed_nodes

    async def _process_single_node(self, node: BaseNode) -> BaseNode:
        try:
            content = node.get_content(MetadataMode.NONE)
            if not content or not content.strip():
                return node

            # Check predicate before applying transform
            if not self.predicate(content):
                logger.debug("Predicate failed, skipping header level reduction")
                return node

            reduced_content = self._reduce_header_levels(content)

            processed_node = node.model_copy()
            processed_node.set_content(reduced_content)

            return processed_node

        except Exception as e:
            logger.error(f"Error reducing header levels: {e}")
            return node

    def _reduce_header_levels(self, content: str) -> str:
        """Reduce all header levels by the specified amount."""
        if self.reduce_by == 0:
            return content

        def replace_header(match: re.Match[str]) -> str:
            header_hashes = match.group(1)
            header_text = match.group(2).strip()
            current_level = len(header_hashes)

            # Calculate new level
            new_level = current_level + self.reduce_by

            # Convert to bold text if exceeds h6
            if new_level > 6:
                return f"**{header_text}**"

            # Return new header with increased level
            return f"{'#' * new_level} {header_text}"

        # Replace all headers using the pattern
        result = self._header_pattern.sub(replace_header, content)

        return result.strip() + "\n\n\n"
