from typing import TYPE_CHECKING
from uuid import uuid4

from injector import inject, singleton
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.llms.llm import ToolSelection

from private_gpt.components.chat.processors.chat_history.documents.document_preprocessor import (
    preprocess_document_history,
)
from private_gpt.components.engines.chat.interceptors.chat_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat.models.chat_interceptor_context import (
    ChatInterceptorContext,
)
from private_gpt.components.engines.chat.models.chat_phase import (
    InterceptorPhase,
)
from private_gpt.events.models import (
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    ToolResultBlock,
    ToolUseBlock,
    to_llama_index_blocks,
)
from private_gpt.server.ingest.convert_service import ConvertService
from private_gpt.settings.settings import Settings

if TYPE_CHECKING:
    from private_gpt.events.models import ResultContentBlockType

DOCUMENT_PROCESSING_TOOL_NAME = "document_preprocessing"


@singleton
class DocumentFilePreprocessingInterceptor(ChatRequestLoopInterceptor):
    """Preprocess DocumentBlock sources by converting file content to plain text."""

    @inject
    def __init__(self, convert_service: ConvertService, settings: Settings) -> None:
        self._convert_service = convert_service
        self._tool_name = DOCUMENT_PROCESSING_TOOL_NAME
        self._preprocess_settings = settings.chat.preprocess.documents

    async def intercept(self, context: ChatInterceptorContext) -> None:
        """Convert document blocks to text before inference."""
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        state = context.state

        tool_ids: dict[int, str] = {}
        completed_tools: list[tuple[str, str | list[ResultContentBlockType], bool]] = []

        async for response in preprocess_document_history(
            chat_history=state.input.request.messages,
            convert_service=self._convert_service,
            max_concurrency=self._preprocess_settings.max_concurrency,
            return_type=self._preprocess_settings.return_type,
        ):
            processing = response.processing_status
            if processing is not None:
                if processing.status == "processing":
                    tool_id = f"tool_{uuid4().hex}"
                    tool_ids[processing.doc_index] = tool_id
                    use_start = RawContentBlockStartEvent(
                        block_id=f"block_{uuid4().hex}",
                        content_block=ToolUseBlock(
                            id=tool_id,
                            name=self._tool_name,
                            input={
                                "type": "document",
                                "index": processing.doc_index,
                                "name": processing.reference,
                            },
                        ),
                    )
                    context.emit_event(use_start)
                    context.emit_event(RawContentBlockStopEvent.from_start(use_start))
                elif processing.status in {"completed", "failed"}:
                    tool_id = tool_ids.get(processing.doc_index, f"tool_{uuid4().hex}")
                    content: str | list[ResultContentBlockType] = (
                        processing.content
                        or processing.error_detail
                        or "There was an error during document processing."
                    )
                    result_start = RawContentBlockStartEvent(
                        block_id=f"block_{uuid4().hex}",
                        content_block=ToolResultBlock(
                            tool_use_id=tool_id,
                            content=content,
                            is_error=processing.status == "failed",
                        ),
                    )
                    context.emit_event(result_start)
                    context.emit_event(
                        RawContentBlockStopEvent.from_start(result_start)
                    )
                    if self._preprocess_settings.return_type == "tool_result":
                        completed_tools.append(
                            (tool_id, content, processing.status == "failed")
                        )

            if response.chat_history is not None:
                state.input.request.messages = response.chat_history

        if self._preprocess_settings.return_type == "tool_result" and completed_tools:
            assistant_msg = ChatMessage(
                role=MessageRole.ASSISTANT,
                content="",
                additional_kwargs={
                    "tool_calls": [
                        ToolSelection(
                            tool_id=tool_id,
                            tool_name=self._tool_name,
                            tool_kwargs={},
                        )
                        for tool_id, _, _ in completed_tools
                    ]
                },
            )
            tool_msgs: list[ChatMessage] = []
            for tool_id, content, _ in completed_tools:
                kwargs = {
                    "tool_call_id": tool_id,
                    "tool_call_name": self._tool_name,
                    "raw_output": content,
                }
                if isinstance(content, str):
                    tool_msgs.append(
                        ChatMessage(
                            role=MessageRole.TOOL,
                            content=content,
                            additional_kwargs=kwargs,
                        )
                    )
                else:
                    tool_msgs.append(
                        ChatMessage(
                            role=MessageRole.TOOL,
                            blocks=to_llama_index_blocks(content),
                            additional_kwargs=kwargs,
                        )
                    )
            state.input.request.messages = [
                *state.input.request.messages,
                assistant_msg,
                *tool_msgs,
            ]

        context.set_state(state)
