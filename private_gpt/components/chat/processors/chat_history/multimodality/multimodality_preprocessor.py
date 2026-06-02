# multimodality_preprocessor.py
from collections.abc import AsyncIterator
from typing import Any

from llama_index.core.base.llms.types import MessageRole, TextBlock
from llama_index.core.llms import LLM, ChatMessage

from private_gpt.components.chat.processors.chat_history.multimodality.audio_preprocessor import (
    preprocess_audio_message,
)
from private_gpt.components.chat.processors.chat_history.multimodality.image_preprocessor import (
    preprocess_image_message,
)
from private_gpt.components.chat.processors.chat_history.multimodality.models import (
    MultimodalProcessingResponse,
)


async def preprocess_multimodal_message(
    main_llm: LLM,
    message: ChatMessage,
    image_multimodal_llm: LLM | None = None,
    audio_multimodal_llm: LLM | None = None,
    **kwargs: Any,
) -> AsyncIterator[MultimodalProcessingResponse]:
    final_message: ChatMessage = message.model_copy()
    final_blocks: list[Any] = []

    original_image_message = message.model_copy()
    async for image_response in preprocess_image_message(
        main_llm, final_message, image_multimodal_llm, **kwargs
    ):
        if image_response.processing_status:
            yield MultimodalProcessingResponse(
                processing_status=image_response.processing_status
            )
        if image_response.message and image_response.message != original_image_message:
            final_blocks.extend(image_response.message.blocks)

    if final_blocks:
        final_message.blocks = final_blocks
        final_blocks = []

    original_audio_message = final_message.model_copy()
    async for audio_response in preprocess_audio_message(
        main_llm, final_message, audio_multimodal_llm, **kwargs
    ):
        if audio_response.processing_status:
            yield MultimodalProcessingResponse(
                processing_status=audio_response.processing_status,
            )
        if audio_response.message and audio_response.message != original_audio_message:
            final_blocks.extend(audio_response.message.blocks)

        if final_blocks:
            final_message.blocks = final_blocks

    yield MultimodalProcessingResponse(modified_message=final_message)


async def preprocess_multimodal_history(
    main_llm: LLM,
    chat_history: list[ChatMessage] | None,
    image_multimodal_llm: LLM | None = None,
    audio_multimodal_llm: LLM | None = None,
    **kwargs: Any,
) -> AsyncIterator[MultimodalProcessingResponse]:
    if not chat_history:
        yield MultimodalProcessingResponse(chat_history=chat_history)
        return

    is_first_user_message = True
    preprocessed_history = []

    for message in reversed(chat_history):
        if message.role == MessageRole.USER and is_first_user_message:
            async for response in preprocess_multimodal_message(
                main_llm,
                message,
                image_multimodal_llm=image_multimodal_llm,
                audio_multimodal_llm=audio_multimodal_llm,
                **kwargs,
            ):
                if response.processing_status:
                    yield response
                if response.modified_message:
                    preprocessed_history.append(response.modified_message)

            is_first_user_message = False
        else:
            message.blocks = [
                block for block in message.blocks if isinstance(block, TextBlock)
            ]
            preprocessed_history.append(message)

    final_history = list(reversed(preprocessed_history))
    yield MultimodalProcessingResponse(chat_history=final_history)
