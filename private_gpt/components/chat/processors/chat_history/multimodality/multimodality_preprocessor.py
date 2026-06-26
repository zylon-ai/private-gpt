import asyncio
from collections.abc import AsyncIterator
from typing import Any, Literal

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
    MultimodalProcessingStatus,
)
from private_gpt.components.chat.processors.chat_history.multimodality.utils import (
    extract_audio_blocks,
    extract_image_blocks,
    requires_audio_preprocessing,
    requires_image_preprocessing,
)
from private_gpt.events.event_errors import Errors


async def _collect_image_response(
    main_llm: LLM,
    message: ChatMessage,
    image_multimodal_llm: LLM | None,
    return_type: Literal["user_message", "tool_result"] = "user_message",
    **kwargs: Any,
) -> tuple[list[MultimodalProcessingStatus], ChatMessage]:
    statuses: list[MultimodalProcessingStatus] = []
    final_message = message
    async for resp in preprocess_image_message(
        main_llm, message, image_multimodal_llm, return_type=return_type, **kwargs
    ):
        if resp.processing_status is not None:
            statuses.append(resp.processing_status)
        if resp.message is not None:
            final_message = resp.message
    return statuses, final_message


async def _collect_audio_response(
    main_llm: LLM,
    message: ChatMessage,
    audio_multimodal_llm: LLM | None,
    return_type: Literal["user_message", "tool_result"] = "user_message",
    **kwargs: Any,
) -> tuple[list[MultimodalProcessingStatus], ChatMessage]:
    statuses: list[MultimodalProcessingStatus] = []
    final_message = message
    async for resp in preprocess_audio_message(
        main_llm, message, audio_multimodal_llm, return_type=return_type, **kwargs
    ):
        if resp.processing_status is not None:
            statuses.append(resp.processing_status)
        if resp.message is not None:
            final_message = resp.message
    return statuses, final_message


async def preprocess_multimodal_message(
    main_llm: LLM,
    message: ChatMessage,
    image_multimodal_llm: LLM | None = None,
    audio_multimodal_llm: LLM | None = None,
    max_concurrency: int | None = None,
    return_type: Literal["user_message", "tool_result"] = "user_message",
    **kwargs: Any,
) -> AsyncIterator[MultimodalProcessingResponse]:
    """Process image and audio blocks in the message in parallel.

    max_concurrency limits how many modalities run simultaneously.
    -1 (default) means unlimited.
    """
    # Pre-check which modalities will actually run so we can emit "processing"
    # events before the parallel tasks start.
    needs_image = (
        requires_image_preprocessing(main_llm, image_multimodal_llm)
        and bool(extract_image_blocks(message))
        and image_multimodal_llm is not None
    )
    needs_audio = (
        requires_audio_preprocessing(main_llm, audio_multimodal_llm)
        and bool(extract_audio_blocks(message))
        and audio_multimodal_llm is not None
    )

    if needs_image:
        yield MultimodalProcessingResponse(
            processing_status=MultimodalProcessingStatus(
                status="processing", type="image"
            )
        )
    if needs_audio:
        yield MultimodalProcessingResponse(
            processing_status=MultimodalProcessingStatus(
                status="processing", type="audio"
            )
        )

    if not needs_image and not needs_audio:
        yield MultimodalProcessingResponse(modified_message=message)
        return

    semaphore = (
        asyncio.Semaphore(max_concurrency)
        if max_concurrency and max_concurrency > 0
        else None
    )

    async def _bounded_image() -> tuple[list[MultimodalProcessingStatus], ChatMessage]:
        if semaphore is not None:
            async with semaphore:
                return await _collect_image_response(
                    main_llm,
                    message,
                    image_multimodal_llm,
                    return_type=return_type,
                    **kwargs,
                )
        return await _collect_image_response(
            main_llm, message, image_multimodal_llm, return_type=return_type, **kwargs
        )

    async def _bounded_audio() -> tuple[list[MultimodalProcessingStatus], ChatMessage]:
        if semaphore is not None:
            async with semaphore:
                return await _collect_audio_response(
                    main_llm,
                    message,
                    audio_multimodal_llm,
                    return_type=return_type,
                    **kwargs,
                )
        return await _collect_audio_response(
            main_llm, message, audio_multimodal_llm, return_type=return_type, **kwargs
        )

    # Run image and audio preprocessing concurrently (bounded by semaphore when set).
    results = await asyncio.gather(
        _bounded_image(),
        _bounded_audio(),
        return_exceptions=True,
    )
    image_result, audio_result = results

    # RequestTooLarge must abort the request — re-raise the first one found.
    for result in results:
        if isinstance(result, Errors.RequestTooLarge):
            raise result

    image_statuses, image_msg = (
        image_result  # type: ignore[misc]
        if not isinstance(image_result, Exception)
        else ([], message)
    )
    audio_statuses, audio_msg = (
        audio_result  # type: ignore[misc]
        if not isinstance(audio_result, Exception)
        else ([], message)
    )

    for status in image_statuses:
        if status.status in {"completed", "failed"}:
            yield MultimodalProcessingResponse(processing_status=status)
    for status in audio_statuses:
        if status.status in {"completed", "failed"}:
            yield MultimodalProcessingResponse(processing_status=status)

    final_blocks = list(message.blocks)

    if image_statuses:
        image_blocks = extract_image_blocks(message)
        final_blocks = [b for b in final_blocks if b not in image_blocks]
        if return_type == "user_message" and image_msg.blocks:
            final_blocks.append(image_msg.blocks[-1])

    if audio_statuses:
        audio_blocks = extract_audio_blocks(message)
        final_blocks = [b for b in final_blocks if b not in audio_blocks]
        if return_type == "user_message" and audio_msg.blocks:
            final_blocks.append(audio_msg.blocks[-1])

    yield MultimodalProcessingResponse(
        modified_message=ChatMessage(role=message.role, blocks=final_blocks)
    )


async def preprocess_multimodal_history(
    main_llm: LLM,
    chat_history: list[ChatMessage] | None,
    image_multimodal_llm: LLM | None = None,
    audio_multimodal_llm: LLM | None = None,
    max_concurrency: int | None = None,
    return_type: Literal["user_message", "tool_result"] = "user_message",
    **kwargs: Any,
) -> AsyncIterator[MultimodalProcessingResponse]:
    if not chat_history:
        yield MultimodalProcessingResponse(chat_history=chat_history)
        return

    if chat_history[-1].role != MessageRole.USER:
        yield MultimodalProcessingResponse(chat_history=chat_history)
        return

    is_last_user_message = True
    preprocessed_history = []

    for message in reversed(chat_history):
        if message.role == MessageRole.USER and is_last_user_message:
            async for response in preprocess_multimodal_message(
                main_llm,
                message,
                image_multimodal_llm=image_multimodal_llm,
                audio_multimodal_llm=audio_multimodal_llm,
                max_concurrency=max_concurrency,
                return_type=return_type,
                **kwargs,
            ):
                if response.processing_status:
                    yield response
                if response.modified_message:
                    preprocessed_history.append(response.modified_message)

            is_last_user_message = False
        else:
            message.blocks = [
                block for block in message.blocks if isinstance(block, TextBlock)
            ]
            preprocessed_history.append(message)

    final_history = list(reversed(preprocessed_history))
    yield MultimodalProcessingResponse(chat_history=final_history)
