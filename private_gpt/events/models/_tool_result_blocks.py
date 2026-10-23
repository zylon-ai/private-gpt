from collections.abc import Sequence
from typing import Annotated, Literal, Protocol, Self, runtime_checkable

from pydantic import Field

from private_gpt.events.models._base import (
    CacheableContentBlock,
    StandardContentProtocol,
)
from private_gpt.events.models._content_blocks import (
    BashCodeExecutionResultBlock,
    CodeExecutionToolResultErrorBlock,
    ResultContentBlockType,
    SourceBlock,
    TextBlock,
    TextEditorCodeExecutionCreateResultBlock,
    TextEditorCodeExecutionStrReplaceResultBlock,
    TextEditorCodeExecutionViewResultBlock,
    WebFetchResultBlock,
    WebFetchToolResultErrorBlock,
    WebSearchResultBlock,
    WebSearchToolResultError,
)


@runtime_checkable
class Renderable(Protocol):
    def render(self) -> str: ...


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

    def for_response_mode(
        self, response_mode: Literal["anthropic", "zylon"]
    ) -> Self | None:
        if isinstance(self.content, str):
            return self
        pruned = [
            b
            for block in self.content
            if (b := block.for_response_mode(response_mode)) is not None
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

    def render(self) -> str:
        """Convert the tool result to a plain-text representation."""
        if isinstance(self.content, str):
            return self.content
        if isinstance(self.content, Sequence):
            return "\n".join(
                block.render() if isinstance(block, Renderable) else str(block)
                for block in self.content
            )
        return str(self.content)

    def for_response_mode(
        self, response_mode: Literal["anthropic", "zylon"]
    ) -> "ToolResultBlock | None":
        if isinstance(self.content, str):
            return self
        if not isinstance(self.content, Sequence):
            block = self.content.for_response_mode(response_mode)
            if block is not None:
                self.content = block
                return self
            return None
        pruned = [
            b
            for block in self.content
            if (b := block.for_response_mode(response_mode)) is not None
        ]
        if pruned:
            self.content = pruned
            return self
        return None


class WebSearchToolResultBlock(ServerToolResultBlock):
    """Anthropic-shaped result for an internally-executed web_search call.

    ``anthropic`` mode: emitted as-is (matches Anthropic wire format).
    ``zylon`` mode: downgrades to ToolResultBlock + SourceBlock/TextBlock entries.
    """

    type: Literal["web_search_tool_result"] = "web_search_tool_result"
    content: list[WebSearchResultBlock] | WebSearchToolResultError
    is_error: bool = Field(default=False, exclude=True)

    def render(self) -> str:
        if isinstance(self.content, WebSearchToolResultError):
            return self.content.render()
        parts = []
        for result in self.content:
            entry = f"{result.title}\n"
            entry += f"Description: {result.description or ''}\n"
            entry += f"URL: {result.url}\n"
            text = result.content or result.encrypted_content
            if text:
                entry += f"Content: {text}\n"
            parts.append(entry)
        return "\n".join(parts)

    def for_response_mode(
        self, response_mode: Literal["anthropic", "zylon"]
    ) -> "Self | ToolResultBlock | None":
        if response_mode == "anthropic":
            if (
                isinstance(self.content, WebSearchToolResultError)
                and self.content.detail is not None
            ):
                stripped = self.content.model_copy(update={"detail": None})
                return self.model_copy(update={"content": stripped})
            return self
        if isinstance(self.content, list):
            from private_gpt.components.chunk.models import Website

            websites = [Website.from_web_search_result(r) for r in self.content]
            zylon_content: list[ResultContentBlockType] = [
                SourceBlock.from_sources(websites),
                *[
                    TextBlock(text=r.content or r.encrypted_content)
                    for r in self.content
                ],
            ]
            is_err = False
        else:
            zylon_content = [TextBlock(text=self.content.render())]
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
    content: WebFetchResultBlock | WebFetchToolResultErrorBlock
    is_error: bool = Field(default=False, exclude=True)

    def render(self) -> str:
        return self.content.render()

    def for_response_mode(
        self, response_mode: Literal["anthropic", "zylon"]
    ) -> "Self | ToolResultBlock | None":
        if response_mode == "anthropic":
            if (
                isinstance(self.content, WebFetchToolResultErrorBlock)
                and self.content.detail is not None
            ):
                stripped = self.content.model_copy(update={"detail": None})
                return self.model_copy(update={"content": stripped})
            return self
        if isinstance(self.content, WebFetchResultBlock):
            text = self.content.markdown or ""
            zylon_content_: list[ResultContentBlockType] = [TextBlock(text=text)]
            is_err = False
        else:
            zylon_content_ = [TextBlock(text=self.content.render())]
            is_err = True
        return ToolResultBlock(
            tool_use_id=self.tool_use_id,
            content=zylon_content_,
            is_error=is_err,
        )


class BashCodeExecutionToolResultBlock(ServerToolResultBlock):
    type: Literal["bash_code_execution_tool_result"] = "bash_code_execution_tool_result"
    content: BashCodeExecutionResultBlock | CodeExecutionToolResultErrorBlock
    is_error: bool = Field(default=False, exclude=True)

    def render(self) -> str:
        return self.content.render()


class TextEditorCodeExecutionToolResultBlock(ServerToolResultBlock):
    type: Literal["text_editor_code_execution_tool_result"] = (
        "text_editor_code_execution_tool_result"
    )
    content: TextEditorCodeExecutionResultContent
    is_error: bool = Field(default=False, exclude=True)

    def render(self) -> str:
        return self.content.render()


ServerToolResultBlockType = (
    ServerToolResultBlock
    | BashCodeExecutionToolResultBlock
    | TextEditorCodeExecutionToolResultBlock
    | WebSearchToolResultBlock
    | WebFetchToolResultBlock
)

ContentBlockType = ResultContentBlockType | ToolResultBlock | ServerToolResultBlockType
