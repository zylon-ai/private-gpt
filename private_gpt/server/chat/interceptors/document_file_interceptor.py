from uuid import uuid4

from injector import inject, singleton

from private_gpt.components.chat.processors.chat_history.documents.document_preprocessor import (
    preprocess_document_message,
)
from private_gpt.components.engines.chat_loop.interceptors.chat_loop_interceptor import (
    ChatRequestLoopInterceptor,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_interceptor_context import (
    ChatLoopInterceptorContext,
)
from private_gpt.components.engines.chat_loop.models.chat_loop_phase import (
    InterceptorPhase,
)
from private_gpt.events.models import (
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    ToolResultBlock,
    ToolUseBlock,
)
from private_gpt.server.ingest.convert_service import ConvertService

DOCUMENT_PROCESSING_TOOL_NAME = "document_preprocessing"


@singleton
class DocumentFilePreprocessingInterceptor(ChatRequestLoopInterceptor):
    """Preprocess DocumentBlock sources by converting file content to plain text."""

    @inject
    def __init__(self, convert_service: ConvertService) -> None:
        self._convert_service = convert_service
        self._tool_name = DOCUMENT_PROCESSING_TOOL_NAME

    async def intercept(self, context: ChatLoopInterceptorContext) -> None:
        """Convert document blocks to text before inference."""
        if context.phase != InterceptorPhase.BEFORE_ITERATION:
            return

        state = context.state
        updated_messages = list(state.input.request.messages)

        for idx, message in enumerate(updated_messages):
            async for response in preprocess_document_message(
                message=message,
                convert_service=self._convert_service,
            ):
                processing = response.processing_status
                if processing is not None:
                    tool_id = f"tool_{uuid4().hex}"
                    if processing.status == "processing":
                        use_start = RawContentBlockStartEvent(
                            block_id=f"block_{uuid4().hex}",
                            content_block=ToolUseBlock(
                                id=tool_id,
                                name=self._tool_name,
                                input={"type": "document"},
                            ),
                        )
                        context.emit_event(use_start)
                        context.emit_event(
                            RawContentBlockStopEvent.from_start(use_start)
                        )
                    elif processing.status in {"completed", "failed"}:
                        result_start = RawContentBlockStartEvent(
                            block_id=f"block_{uuid4().hex}",
                            content_block=ToolResultBlock(
                                tool_use_id=tool_id,
                                content=(
                                    processing.content
                                    or processing.error_detail
                                    or "There was an error during document processing."
                                ),
                                is_error=processing.status == "failed",
                            ),
                        )
                        context.emit_event(result_start)
                        context.emit_event(
                            RawContentBlockStopEvent.from_start(result_start)
                        )

                if response.message is not None:
                    updated_messages[idx] = response.message

        new_state = state.model_copy(deep=True)
        new_state.input.request.messages = updated_messages
        context.set_state(new_state)
