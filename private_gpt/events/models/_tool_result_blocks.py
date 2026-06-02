from collections.abc import Sequence
from typing import Annotated, Literal, Self

from pydantic import Field

from private_gpt.events.models._base import (
    CacheableContentBlock,
    StandardContentProtocol,
)
from private_gpt.events.models._content_blocks import ResultContentBlockType


class ToolReferenceBlock(CacheableContentBlock, StandardContentProtocol):
    """Reference to a tool name used inside tool_result payloads."""

    type: Literal["tool_reference"] = Field(default="tool_reference")
    tool_name: str = Field(
        min_length=1,
        max_length=256,
        pattern=r"^[a-zA-Z0-9_-]{1,256}$",
        description="Tool name reference.",
    )


ToolResultContentBlockType = ResultContentBlockType | ToolReferenceBlock


class ToolResultBlock(CacheableContentBlock, StandardContentProtocol):
    """Result produced by a prior tool_use block."""

    type: Literal["tool_result"] = Field(default="tool_result")
    tool_use_id: str = Field(
        description="ID of the ToolUseBlock this result answers",
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    content: str | Sequence[
        Annotated[ToolResultContentBlockType, Field(discriminator="type")]
    ] = Field(default="", description="Tool execution result")
    is_error: bool = Field(
        default=False, description="Whether the tool result indicates an error."
    )

    def prune_content_block_by_response_mode(
        self, response_mode: Literal["anthropic", "zylon"]
    ) -> Self | None:
        if isinstance(self.content, str):
            return self
        pruned = [
            b
            for block in self.content
            if (b := block.prune_content_block_by_response_mode(response_mode))
            is not None
        ]
        if pruned:
            self.content = pruned
            return self
        return None


ContentBlockType = ResultContentBlockType | ToolResultBlock
