from collections.abc import Sequence
from typing import Any

from llama_index.core.schema import BaseNode, MetadataMode, TransformComponent

from private_gpt.components.ingest.metadata_helper import MetadataNode
from private_gpt.components.llm.llm_helper import TokenizerFn, get_tokenizer
from private_gpt.components.llm.tokenizers.tokenizer_base import (
    AsyncTokenizerBase,
    TokenizerBase,
)
from private_gpt.components.readers.nodes.tree_node import TreeMetadataMode, TreeNode


class IncludeTokenCountIntoNodesTransform(TransformComponent):
    """Include token length in the nodes.

    With a ``batch_tokenizer`` the nodes are tokenized as one batch
    (``aencode_batch`` / ``encode_batch``) instead of one blocking call per node:
    remote tokenizers made a 4,000-node document take ~20 minutes.
    """

    tokenizer: TokenizerFn | None
    batch_tokenizer: TokenizerBase | None = None

    @classmethod
    def from_defaults(
        cls,
        tokenizer: TokenizerFn | None = None,
        batch_tokenizer: TokenizerBase | None = None,
    ) -> "IncludeTokenCountIntoNodesTransform":
        if tokenizer is None and batch_tokenizer is None:
            batch_tokenizer = _default_tokenizer()
            tokenizer = get_tokenizer()
        return cls(tokenizer=tokenizer, batch_tokenizer=batch_tokenizer)

    @staticmethod
    def _content(node: BaseNode) -> str:
        return (
            node.get_content(TreeMetadataMode.RAG)
            if isinstance(node, TreeNode)
            else node.get_content(MetadataMode.NONE)
        )

    @staticmethod
    def _set_count(node: BaseNode, count: int) -> None:
        node.metadata[MetadataNode.TOKEN_COUNT.value] = count
        node.excluded_llm_metadata_keys.append(MetadataNode.TOKEN_COUNT.value)
        node.excluded_embed_metadata_keys.append(MetadataNode.TOKEN_COUNT.value)

    @classmethod
    def _group_by_content(cls, nodes: Sequence[BaseNode]) -> dict[str, list[BaseNode]]:
        """Nodes per content; empty ones are skipped to avoid a zero count.

        Identical texts (repeated headers, table rows) are tokenized once.
        """
        groups: dict[str, list[BaseNode]] = {}
        for node in nodes:
            if content := cls._content(node):
                groups.setdefault(content, []).append(node)
        return groups

    def _apply(
        self, groups: dict[str, list[BaseNode]], token_ids: list[list[int]]
    ) -> None:
        for group, ids in zip(groups.values(), token_ids, strict=True):
            for node in group:
                self._set_count(node, len(ids))

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        groups = self._group_by_content(nodes)
        if self.batch_tokenizer is not None:
            self._apply(groups, self.batch_tokenizer.encode_batch(list(groups)))
        elif self.tokenizer is not None:
            self._apply(groups, [list(self.tokenizer(c)) for c in groups])
        return nodes

    async def acall(
        self, nodes: Sequence[BaseNode], **kwargs: Any
    ) -> Sequence[BaseNode]:
        if not isinstance(self.batch_tokenizer, AsyncTokenizerBase):
            return self.__call__(nodes, **kwargs)
        groups = self._group_by_content(nodes)
        self._apply(groups, await self.batch_tokenizer.aencode_batch(list(groups)))
        return nodes


def _default_tokenizer() -> TokenizerBase | None:
    from private_gpt.components.llm.llm_component import LLMComponent
    from private_gpt.di import get_global_injector

    return get_global_injector().get(LLMComponent).tokenizer
