from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole

if TYPE_CHECKING:
    from private_gpt.components.llm.tokenizers.tokenizer_base import TokenizerBase


class TextParserBase:
    """Abstract text parser class that should not be used directly.

    Provided and methods should be used in derived classes.

    It is used to extract text content from the model output.
    """

    def __init__(self, tokenizer: TokenizerBase, **kwargs: Any) -> None:
        self.model_tokenizer = tokenizer

    @classmethod
    def from_prototype(
        cls,
        prototype: TextParserBase,
        **kwargs: Any,
    ) -> TextParserBase:
        """Create a new instance of the ToolParser class."""
        return prototype.__class__(prototype.model_tokenizer, **kwargs)

    def extract_text_content(
        self,
        model_output: str,
    ) -> str:
        """Extract text content from a complete model-generated string.

        Used for non-streaming responses where we have the entire model response
        available before sending to the client.

        Parameters:
        model_output: str
            The model-generated string to extract text content from.

        request: ChatCompletionRequest
            The request object that was used to generate the model_output.

        Returns:
        str
            The extracted text content.
        """
        return model_output

    def extract_text_content_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
    ) -> ChatResponse | None:
        """Method to extract text content from a delta message.

        Instance method that should be implemented for extracting text
        from an incomplete response; for use when handling text calls and
        streaming. Has to be an instance method because  it requires state -
        the current tokens/diffs, but also the information about what has
        previously been parsed and extracted (see constructor)
        """
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=current_text),
            delta=delta_text,
            raw=current_text,
        )

    def close(self) -> None:
        """Close the reasoning parser."""
        pass
