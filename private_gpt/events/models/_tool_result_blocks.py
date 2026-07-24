from collections.abc import Sequence
from typing import Annotated, Literal, Self

from pydantic import Field

from private_gpt.events.models._base import (
    CacheableContentBlock,
    StandardContentProtocol,
)
from private_gpt.events.models._content_blocks import (
    DocumentBlock,
    ResultContentBlockType,
    SourceBlock,
    TextBlock,
    WebFetchResultBlock,
    WebSearchResultBlock,

    BashCodeExecutionResultBlock,
    CodeExecutionToolResultErrorBlock,
    ResultContentBlockType,
    TextEditorCodeExecutionCreateResultBlock,
    TextEditorCodeExecutionStrReplaceResultBlock,
    TextEditorCodeExecutionViewResultBlock,
)


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
    """Shared interface for results produced by tool-use blocks."""

    type: Literal["tool_result"] = Field(default="tool_result")
    tool_use_id: str = Field(
        description="ID of the ToolUseBlock this result answers",
        pattern=r"^[a-zA-Z0-9_-]+$",
    )
    content: (
        str
        | Sequence[Annotated[ToolResultContentBlockType, Field(discriminator="type")]]
    ) = Field(default="", description="Tool execution result")
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


class ClientToolResultBlock(ToolResultBlock):
    """Result supplied for a client-executed tool call."""

    type: Literal["tool_result"] = Field(default="tool_result")


TextEditorCodeExecutionResultContent = Annotated[
    TextEditorCodeExecutionViewResultBlock
    | TextEditorCodeExecutionCreateResultBlock
    | TextEditorCodeExecutionStrReplaceResultBlock
    | CodeExecutionToolResultErrorBlock,
    Field(discriminator="type"),
]


class ServerToolResultBlock(ToolResultBlock):
    """Default result produced by a server-executed tool."""

    type: Literal["server_tool_result"] = Field(default="server_tool_result")



class WebSearchToolResultBlock(ServerToolResultBlock):
    """Anthropic-shaped result for an internally-executed web_search call.

    ``anthropic`` mode: emitted as-is (matches Anthropic wire format).
    ``zylon`` mode: downgrades to ToolResultBlock + SourceBlock/TextBlock entries.
    """

    type: Literal["web_search_tool_result"] = "web_search_tool_result"
    tool_use_id: str
    content: list[WebSearchResultBlock] | CodeExecutionToolResultErrorBlock
    is_error: bool = Field(default=False, exclude=True)

    def prune_content_block_by_response_mode(
        self, response_mode: Literal["anthropic", "zylon"]
    ) -> "Self | ToolResultBlock | None":
        if response_mode == "anthropic":
            return self
        if isinstance(self.content, list):
            from private_gpt.components.chunk.models import Website

            websites = [Website.from_web_search_result(r) for r in self.content]
            zylon_content: list[ResultContentBlockType] = [
                SourceBlock.from_sources(websites),
                *[TextBlock(text=r.content or r.encrypted_content) for r in self.content],
            ]
            is_err = False
        else:
            zylon_content = [TextBlock(text=f"Web search error: {self.content.error_code}")]
            is_err = True
        return ToolResultBlock(
            tool_use_id=self.tool_use_id,
            content=zylon_content,
            is_error=is_err,
        )


class WebFetchToolResultBlock(ServerToolResultBlock):
    """Anthropic-shaped result for an internally-executed web_fetch call.

    ``anthropic`` mode: emitted as-is.
    ``zylon`` mode: downgrades to ToolResultBlock with a TextBlock of the markdown.
    """

    type: Literal["web_fetch_tool_result"] = "web_fetch_tool_result"
    tool_use_id: str
    content: WebFetchResultBlock | CodeExecutionToolResultErrorBlock
    is_error: bool = Field(default=False, exclude=True)

    def prune_content_block_by_response_mode(
        self, response_mode: Literal["anthropic", "zylon"]
    ) -> "Self | ToolResultBlock | None":
        if response_mode == "anthropic":
            return self
        if isinstance(self.content, WebFetchResultBlock):
            src = self.content.content.source
            text = self.content.markdown or (
                src.data if isinstance(src, DocumentBlock.Base64Source) else src.url
            )
            zylon_content_: list[ResultContentBlockType] = [TextBlock(text=text or "")]
            is_err = False
        else:
            zylon_content_ = [TextBlock(text=f"Web fetch error: {self.content.error_code}")]
            is_err = True
        return ToolResultBlock(
            tool_use_id=self.tool_use_id,
            content=zylon_content_,
            is_error=is_err,
        )


class BashCodeExecutionToolResultBlock(ServerToolResultBlock):
    type: Literal["bash_code_execution_tool_result"] = "bash_code_execution_tool_result"
    tool_use_id: str
    content: BashCodeExecutionResultBlock | CodeExecutionToolResultErrorBlock
    is_error: bool = Field(default=False, exclude=True)


class TextEditorCodeExecutionToolResultBlock(ServerToolResultBlock):
    type: Literal["text_editor_code_execution_tool_result"] = (
        "text_editor_code_execution_tool_result"
    )
    tool_use_id: str
    content: TextEditorCodeExecutionResultContent
    is_error: bool = Field(default=False, exclude=True)


ServerToolResultBlockType = (
    ServerToolResultBlock
    | BashCodeExecutionToolResultBlock
    | TextEditorCodeExecutionToolResultBlock
    | WebSearchToolResultBlock
    | WebFetchToolResultBlock
)

ContentBlockType = ResultContentBlockType | ToolResultBlock | ServerToolResultBlockType
