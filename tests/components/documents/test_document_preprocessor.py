import base64
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.base.llms.types import TextBlock
from llama_index.core.llms import ChatMessage, MessageRole

from private_gpt.components.chat.processors.chat_history.documents.document_preprocessor import (
    DocumentProcessingResponse,
    preprocess_document_message,
)
from private_gpt.events.models import DocumentBlock
from private_gpt.events.models import TextBlock as PGTextBlock
from private_gpt.server.ingest.convert_service import ConvertService


def _mock_convert_service(text: str = "extracted text") -> MagicMock:
    mock = MagicMock(spec=ConvertService)
    mock.bytes_to_text.return_value = text
    return mock


def _msg(*doc_blocks: DocumentBlock, text: str | None = None) -> ChatMessage:
    """Build a ChatMessage with DocumentBlocks stored in additional_kwargs."""
    blocks = [TextBlock(text=text)] if text else []
    return ChatMessage(
        role=MessageRole.USER,
        blocks=blocks,
        additional_kwargs={"document": list(doc_blocks)} if doc_blocks else {},
    )


async def _collect(gen) -> list[DocumentProcessingResponse]:
    return [r async for r in gen]


@pytest.fixture
def convert_service() -> MagicMock:
    return _mock_convert_service()


@pytest.fixture
def text_message() -> ChatMessage:
    return ChatMessage(
        role=MessageRole.USER,
        blocks=[TextBlock(text="Hello")],
    )


@pytest.fixture
def plain_text_message() -> ChatMessage:
    return _msg(
        DocumentBlock(
            source=DocumentBlock.PlainTextSource(
                type="text", data="plain text content", media_type="text/plain"
            )
        ),
        text="Here is a doc",
    )


@pytest.fixture
def base64_message() -> ChatMessage:
    return _msg(
        DocumentBlock(
            source=DocumentBlock.Base64Source(
                type="base64",
                data=base64.b64encode(b"%PDF-1.4 fake").decode(),
                media_type="application/pdf",
            )
        )
    )


@pytest.fixture
def url_message() -> ChatMessage:
    return _msg(
        DocumentBlock(
            source=DocumentBlock.URLDocumentSource(
                type="url", url="https://example.com/doc.pdf"
            )
        )
    )


class TestDocumentPreprocessing:
    async def test_text_only_passes_through(
        self, text_message: ChatMessage, convert_service: MagicMock
    ) -> None:
        responses = await _collect(
            preprocess_document_message(text_message, convert_service)
        )
        assert len(responses) == 1
        assert responses[0].message is text_message
        assert responses[0].processing_status is None

    async def test_plain_text_source_extracted(
        self, plain_text_message: ChatMessage, convert_service: MagicMock
    ) -> None:
        responses = await _collect(
            preprocess_document_message(plain_text_message, convert_service)
        )
        message = responses[-1].message
        assert message is not None
        assert any(
            isinstance(b, TextBlock) and "plain text content" in b.text
            for b in message.blocks
        )

    async def test_content_source_str_extracted(
        self, convert_service: MagicMock
    ) -> None:
        message = _msg(
            DocumentBlock(
                source=DocumentBlock.ContentSource(
                    type="content", content="content source text"
                )
            )
        )
        responses = await _collect(
            preprocess_document_message(message, convert_service)
        )
        result = responses[-1].message
        assert result is not None
        assert any(
            isinstance(b, TextBlock) and "content source text" in b.text
            for b in result.blocks
        )

    async def test_content_source_list_extracted(
        self, convert_service: MagicMock
    ) -> None:
        message = _msg(
            DocumentBlock(
                source=DocumentBlock.ContentSource(
                    type="content",
                    content=[PGTextBlock(text="block one")],
                )
            )
        )
        responses = await _collect(
            preprocess_document_message(message, convert_service)
        )
        result = responses[-1].message
        assert result is not None
        assert any(
            isinstance(b, TextBlock) and "block one" in b.text for b in result.blocks
        )

    async def test_base64_source_calls_bytes_to_text(
        self, base64_message: ChatMessage, convert_service: MagicMock
    ) -> None:
        responses = await _collect(
            preprocess_document_message(base64_message, convert_service)
        )
        convert_service.bytes_to_text.assert_called_once()
        _, ext = convert_service.bytes_to_text.call_args[0]
        assert ext == ".pdf"
        message = responses[-1].message
        assert message is not None
        assert any(
            isinstance(b, TextBlock) and "extracted text" in b.text
            for b in message.blocks
        )

    async def test_base64_source_strips_data_uri_prefix(
        self, convert_service: MagicMock
    ) -> None:
        raw = b"%PDF-1.4 fake"
        encoded = base64.b64encode(raw).decode()
        message = _msg(
            DocumentBlock(
                source=DocumentBlock.Base64Source(
                    type="base64",
                    data=f"data:application/pdf;base64,{encoded}",
                    media_type="application/pdf",
                )
            )
        )
        await _collect(preprocess_document_message(message, convert_service))
        convert_service.bytes_to_text.assert_called_once()
        decoded_bytes, ext = convert_service.bytes_to_text.call_args[0]
        assert decoded_bytes == raw
        assert ext == ".pdf"

    async def test_base64_non_pdf_extension_derived_from_media_type(
        self, convert_service: MagicMock
    ) -> None:
        message = _msg(
            DocumentBlock(
                source=DocumentBlock.Base64Source(
                    type="base64",
                    data=base64.b64encode(b"fake docx").decode(),
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            )
        )
        await _collect(preprocess_document_message(message, convert_service))
        _, ext = convert_service.bytes_to_text.call_args[0]
        assert ext == ".docx"

    async def test_url_source_calls_bytes_to_text(
        self, url_message: ChatMessage, convert_service: MagicMock
    ) -> None:
        with patch.object(
            DocumentBlock.URLDocumentSource, "to_bytes", return_value=b"pdf bytes"
        ):
            responses = await _collect(
                preprocess_document_message(url_message, convert_service)
            )
        convert_service.bytes_to_text.assert_called_once_with(b"pdf bytes", ".pdf")
        assert responses[-1].message is not None

    async def test_non_document_blocks_preserved(
        self, plain_text_message: ChatMessage, convert_service: MagicMock
    ) -> None:
        responses = await _collect(
            preprocess_document_message(plain_text_message, convert_service)
        )
        message = responses[-1].message
        assert message is not None
        assert any(
            isinstance(b, TextBlock) and b.text == "Here is a doc"
            for b in message.blocks
        )

    async def test_document_removed_from_additional_kwargs(
        self, plain_text_message: ChatMessage, convert_service: MagicMock
    ) -> None:
        responses = await _collect(
            preprocess_document_message(plain_text_message, convert_service)
        )
        message = responses[-1].message
        assert message is not None
        assert "document" not in message.additional_kwargs

    async def test_title_and_context_prepended(
        self, convert_service: MagicMock
    ) -> None:
        message = _msg(
            DocumentBlock(
                source=DocumentBlock.PlainTextSource(
                    type="text", data="body text", media_type="text/plain"
                ),
                title="My Doc",
                context="some context",
            )
        )
        responses = await _collect(
            preprocess_document_message(message, convert_service)
        )
        result = responses[-1].message
        assert result is not None
        text_block = next(b for b in result.blocks if isinstance(b, TextBlock))
        assert "Title: My Doc" in text_block.text
        assert "Context: some context" in text_block.text
        assert "body text" in text_block.text

    async def test_failed_conversion_yields_fallback(
        self, base64_message: ChatMessage
    ) -> None:
        failing_service = MagicMock(spec=ConvertService)
        failing_service.bytes_to_text.side_effect = RuntimeError("parse error")
        responses = await _collect(
            preprocess_document_message(base64_message, failing_service)
        )
        message = responses[-1].message
        assert message is not None
        assert any(
            isinstance(b, TextBlock) and "document processing failed" in b.text
            for b in message.blocks
        )

    async def test_message_role_preserved(
        self, plain_text_message: ChatMessage, convert_service: MagicMock
    ) -> None:
        responses = await _collect(
            preprocess_document_message(plain_text_message, convert_service)
        )
        assert responses[-1].message.role == MessageRole.USER


class TestDocumentProcessingStatus:
    async def test_no_status_for_text_only(
        self, text_message: ChatMessage, convert_service: MagicMock
    ) -> None:
        responses = await _collect(
            preprocess_document_message(text_message, convert_service)
        )
        assert all(r.processing_status is None for r in responses)

    async def test_processing_status_emitted_first(
        self, plain_text_message: ChatMessage, convert_service: MagicMock
    ) -> None:
        responses = await _collect(
            preprocess_document_message(plain_text_message, convert_service)
        )
        status_responses = [r for r in responses if r.processing_status is not None]
        assert status_responses[0].processing_status.status == "processing"

    async def test_completed_status_on_success(
        self, plain_text_message: ChatMessage, convert_service: MagicMock
    ) -> None:
        responses = await _collect(
            preprocess_document_message(plain_text_message, convert_service)
        )
        statuses = [r.processing_status for r in responses if r.processing_status]
        assert any(s.status == "completed" for s in statuses)

    async def test_failed_status_on_error(self, base64_message: ChatMessage) -> None:
        failing_service = MagicMock(spec=ConvertService)
        failing_service.bytes_to_text.side_effect = RuntimeError("parse error")
        responses = await _collect(
            preprocess_document_message(base64_message, failing_service)
        )
        statuses = [r.processing_status for r in responses if r.processing_status]
        failed = next(s for s in statuses if s.status == "failed")
        assert "parse error" in (failed.error_detail or "")

    async def test_failed_status_has_error_detail(
        self, base64_message: ChatMessage
    ) -> None:
        failing_service = MagicMock(spec=ConvertService)
        failing_service.bytes_to_text.side_effect = ValueError("bad format")
        responses = await _collect(
            preprocess_document_message(base64_message, failing_service)
        )
        statuses = [r.processing_status for r in responses if r.processing_status]
        failed = next(s for s in statuses if s.status == "failed")
        assert failed.error_detail is not None


class TestMultipleDocuments:
    async def test_multiple_documents_all_converted(
        self, convert_service: MagicMock
    ) -> None:
        message = _msg(
            DocumentBlock(
                source=DocumentBlock.PlainTextSource(
                    type="text", data="doc one", media_type="text/plain"
                )
            ),
            DocumentBlock(
                source=DocumentBlock.PlainTextSource(
                    type="text", data="doc two", media_type="text/plain"
                )
            ),
        )
        responses = await _collect(
            preprocess_document_message(message, convert_service)
        )
        result = responses[-1].message
        assert result is not None
        all_text = " ".join(b.text for b in result.blocks if isinstance(b, TextBlock))
        assert "doc one" in all_text
        assert "doc two" in all_text

    async def test_completed_status_mentions_count(
        self, convert_service: MagicMock
    ) -> None:
        message = _msg(
            DocumentBlock(
                source=DocumentBlock.PlainTextSource(
                    type="text", data="a", media_type="text/plain"
                )
            ),
            DocumentBlock(
                source=DocumentBlock.PlainTextSource(
                    type="text", data="b", media_type="text/plain"
                )
            ),
        )
        responses = await _collect(
            preprocess_document_message(message, convert_service)
        )
        statuses = [r.processing_status for r in responses if r.processing_status]
        completed = next(s for s in statuses if s.status == "completed")
        assert completed.content is not None
        assert "2" in str(completed.content)

    async def test_mixed_text_and_document_blocks(
        self, convert_service: MagicMock
    ) -> None:
        message = _msg(
            DocumentBlock(
                source=DocumentBlock.PlainTextSource(
                    type="text", data="doc content", media_type="text/plain"
                )
            ),
            text="check this",
        )
        responses = await _collect(
            preprocess_document_message(message, convert_service)
        )
        result = responses[-1].message
        assert result is not None
        assert any(
            isinstance(b, TextBlock) and b.text == "check this" for b in result.blocks
        )
        assert any(
            isinstance(b, TextBlock) and "doc content" in b.text for b in result.blocks
        )
