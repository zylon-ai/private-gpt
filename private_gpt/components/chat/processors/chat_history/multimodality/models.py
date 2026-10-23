from typing import Literal

from llama_index.core.base.llms.types import ChatMessage
from pydantic import BaseModel

from private_gpt.events.models._content_blocks import ResultContentBlockType


class MultimodalProcessingStatus(BaseModel):
    status: Literal["processing", "completed", "failed"]
    type: Literal["image", "audio", "video", "other"]
    message: ChatMessage | None = None
    content: str | list[ResultContentBlockType] | None = None
    error_detail: str | None = None


class MultimodalProcessingResponse(BaseModel):
    processing_status: MultimodalProcessingStatus | None = None
    chat_history: list[ChatMessage] | None = None
    modified_message: ChatMessage | None = None
