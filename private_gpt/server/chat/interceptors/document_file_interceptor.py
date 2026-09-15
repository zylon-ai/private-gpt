import logging

from injector import inject, singleton

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
from private_gpt.components.ingestion.ingestion_scheduler import (
    IngestionSchedulerFactory,
)
from private_gpt.server.chat.interceptors.preprocessing_tool_calls import (
    PreprocessingToolCalls,
)
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)

DOCUMENT_PROCESSING_TOOL_NAME = "document_preprocessing"


@singleton
class DocumentFilePreprocessingInterceptor(ChatRequestLoopInterceptor):
    """Preprocess DocumentBlock sources by converting file content to plain text."""

    @inject
    def __init__(
        self, scheduler_factory: IngestionSchedulerFactory, settings: Settings
    ) -> None:
        self._scheduler_factory = scheduler_factory
        self._tool_name = DOCUMENT_PROCESSING_TOOL_NAME
        self._preprocess_settings = settings.chat.preprocess.documents

    async def intercept(self, context: ChatInterceptorContext) -> None:
        """Convert document blocks to text before inference.

        This runs only on the first iteration of a request. Once a document
        block has been converted to plain text it is not present in later
        iterations; later iterations still see the updated message history
        through the document/citation/system-prompt interceptors, which are
        intentionally *not* gated.
        """
        if (
            context.phase != InterceptorPhase.BEFORE_ITERATION
            or context.state.runtime.iteration > 0
        ):
            return

        state = context.state
        tool_calls = PreprocessingToolCalls(
            context,
            tool_name=self._tool_name,
            return_type=self._preprocess_settings.return_type,
            default_error="There was an error during document processing.",
        )

        try:
            async for response in preprocess_document_history(
                chat_history=state.input.request.messages,
                convert_service=self._scheduler_factory.get(),
                max_concurrency=self._preprocess_settings.max_concurrency,
                return_type=self._preprocess_settings.return_type,
            ):
                processing = response.processing_status
                if processing is not None:
                    if processing.status == "processing":
                        tool_calls.start(
                            processing.doc_index,
                            {
                                "type": "document",
                                "index": processing.doc_index,
                                "name": processing.reference,
                            },
                        )
                    elif processing.status in {"completed", "failed"}:
                        tool_calls.finish(
                            processing.doc_index,
                            processing.content,
                            is_error=processing.status == "failed",
                            error_detail=processing.error_detail,
                        )

                if response.chat_history is not None:
                    state.input.request.messages = response.chat_history
        except Exception as exc:
            if not tool_calls.has_pending:
                raise
            logger.exception("Document preprocessing failed; reporting as tool error")
            tool_calls.fail_pending(exc)

        state.input.request.messages = tool_calls.append_tool_messages(
            state.input.request.messages
        )
        context.set_state(state)
