from llama_index.core.base.llms.types import AudioBlock, ImageBlock
from llama_index.core.llms import LLM, ChatMessage


def requires_image_preprocessing(main_llm: LLM, image_llm: LLM | None) -> bool:
    if image_llm is None:
        return True  # No image support available
    return main_llm is not image_llm  # Same object means no preprocessing needed


def requires_audio_preprocessing(main_llm: LLM, audio_llm: LLM | None) -> bool:
    if audio_llm is None:
        return True  # No audio support available
    return main_llm is not audio_llm  # Same object means no preprocessing needed


def extract_image_blocks(message: ChatMessage) -> list[ImageBlock]:
    return [block for block in message.blocks if isinstance(block, ImageBlock)]


def extract_audio_blocks(message: ChatMessage) -> list[AudioBlock]:
    return [block for block in message.blocks if isinstance(block, AudioBlock)]


def remove_multimodal_blocks(message: ChatMessage) -> ChatMessage:
    filtered_blocks = [
        block
        for block in message.blocks
        if not isinstance(block, ImageBlock | AudioBlock)
    ]

    return ChatMessage(
        role=message.role,
        content=message.content,
        blocks=filtered_blocks,
    )
