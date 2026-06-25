from collections.abc import AsyncIterator
from typing import Any, Literal

from llama_index.core.base.llms.types import TextBlock as LITextBlock
from llama_index.core.llms import LLM, ChatMessage
from pydantic import BaseModel

from private_gpt.components.chat.processors.chat_history.multimodality.models import (
    MultimodalProcessingStatus,
)
from private_gpt.components.chat.processors.chat_history.multimodality.utils import (
    extract_audio_blocks,
    requires_audio_preprocessing,
)
from private_gpt.components.multimodality.audio_handler import process_audio_in_message
from private_gpt.events.event_errors import Errors


class AudioProcessingResponse(BaseModel):
    message: ChatMessage | None = None
    processing_status: MultimodalProcessingStatus | None = None


async def preprocess_audio_message(
    main_llm: LLM,
    message: ChatMessage,
    audio_multimodal_llm: LLM | None = None,
    return_type: Literal["user_message", "tool_result"] = "user_message",
    **kwargs: Any,
) -> AsyncIterator[AudioProcessingResponse]:
    """Preprocess a message by extracting insights from audio only.

    Args:
        main_llm: The primary LLM that will process the final message
        message: Original chat message potentially containing audio
        audio_multimodal_llm: LLM capable of processing audio (None if no support)
        return_type: Type of way to return the content
        kwargs: Additional keyword arguments

    Returns:
        Preprocessed message with audio content converted to text descriptions

    Raises:
        ValueError: If audio preprocessing fails or required capabilities are missing
    """
    needs_audio_preprocessing = requires_audio_preprocessing(
        main_llm, audio_multimodal_llm
    )
    if not needs_audio_preprocessing:
        yield AudioProcessingResponse(message=message)
        return

    audio_blocks = extract_audio_blocks(message)
    if not audio_blocks:
        yield AudioProcessingResponse(message=message)
        return

    if audio_multimodal_llm is None:
        raise ValueError("Audio blocks found but no audio-capable LLM provided.")

    event = MultimodalProcessingStatus(status="processing", type="audio")
    yield AudioProcessingResponse(processing_status=event)

    try:
        audio_description = await process_audio_in_message(
            audio_multimodal_llm, message, user_query=message.content, **kwargs
        )

        if not audio_description:
            raise ValueError("Failed to describe audio in the message.")

        event = event.model_copy(
            update={
                "status": "completed",
                "content": audio_description,
            }
        )
        yield AudioProcessingResponse(processing_status=event)
        final_message = (
            "The user has included audios in their message. "
            "We have processed these audios and obtained the following descriptions:\n"
            f"{audio_description}"
        )

    except Errors.RequestTooLarge as e:
        event = event.model_copy(
            update={
                "status": "failed",
                "error_detail": str(e),
            }
        )

        yield AudioProcessingResponse(processing_status=event)

        raise

    except Exception as e:
        event = event.model_copy(
            update={
                "status": "failed",
                "error_detail": str(e),
            }
        )
        yield AudioProcessingResponse(processing_status=event)
        final_message = (
            "The user has included audios in their message. "
            "However, we encountered an error while processing the audio content. "
            "Please inform the user that audio processing failed."
        )

    other_blocks = [block for block in message.blocks if block not in audio_blocks]
    final_blocks = (
        [*other_blocks, LITextBlock(text=final_message)]
        if return_type == "user_message"
        else other_blocks
    )
    yield AudioProcessingResponse(
        message=ChatMessage(role=message.role, blocks=final_blocks)
    )
