from collections.abc import AsyncIterator
from typing import Any

from llama_index.core.base.llms.types import TextBlock as LITextBlock
from llama_index.core.llms import LLM, ChatMessage
from pydantic import BaseModel

from private_gpt.components.chat.processors.chat_history.multimodality.models import (
    MultimodalProcessingStatus,
)
from private_gpt.components.chat.processors.chat_history.multimodality.utils import (
    extract_image_blocks,
    requires_image_preprocessing,
)
from private_gpt.components.multimodality.image_handler import process_images_in_message
from private_gpt.events.event_errors import Errors


class ImageProcessingResponse(BaseModel):
    message: ChatMessage | None = None
    processing_status: MultimodalProcessingStatus | None = None


async def preprocess_image_message(
    main_llm: LLM,
    message: ChatMessage,
    image_multimodal_llm: LLM | None = None,
    **kwargs: Any,
) -> AsyncIterator[ImageProcessingResponse]:
    """Preprocess a message by extracting insights from images only.

    Args:
        main_llm: The primary LLM that will process the final message
        message: Original chat message potentially containing images
        image_multimodal_llm: LLM capable of processing images (None if no support)
        kwargs: Additional keyword arguments

    Returns:
        Preprocessed message with image content converted to text descriptions

    Raises:
        ValueError: If image preprocessing fails or required capabilities are missing
    """
    needs_image_preprocessing = requires_image_preprocessing(
        main_llm, image_multimodal_llm
    )
    if not needs_image_preprocessing:
        yield ImageProcessingResponse(message=message)
        return

    image_blocks = extract_image_blocks(message)
    if not image_blocks:
        yield ImageProcessingResponse(message=message)
        return

    if image_multimodal_llm is None:
        raise ValueError("Image blocks found but no image-capable LLM provided.")

    event = MultimodalProcessingStatus(status="processing", type="image")
    yield ImageProcessingResponse(processing_status=event)

    try:
        image_description = await process_images_in_message(
            image_multimodal_llm, message, user_query=message.content, **kwargs
        )

        if not image_description:
            raise ValueError("Failed to describe images in the message.")

        event = event.model_copy(
            update={
                "status": "completed",
                "content": image_description,
            }
        )
        yield ImageProcessingResponse(processing_status=event)
        final_message = (
            "The user has included images in their message. "
            "We have processed these images and obtained the following descriptions:\n"
            f"{image_description}"
        )

    except Errors.RequestTooLarge as e:
        event = event.model_copy(
            update={
                "status": "failed",
                "error_detail": str(e),
            }
        )
        yield ImageProcessingResponse(processing_status=event)

        raise

    except Exception as e:
        event = event.model_copy(
            update={
                "status": "failed",
                "error_detail": str(e),
            }
        )
        yield ImageProcessingResponse(processing_status=event)
        final_message = (
            "The user has included images in their message. "
            "However, we were unable to process these images due to an error. "
            "Please inform the user that image processing failed."
        )

    other_blocks = [block for block in message.blocks if block not in image_blocks]
    yield ImageProcessingResponse(
        message=ChatMessage(
            role=message.role,
            blocks=[
                *other_blocks,
                LITextBlock(text=final_message),
            ],
        )
    )
