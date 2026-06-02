from typing import TYPE_CHECKING, Any

import pytest
from llama_index.core.base.llms.types import (
    AudioBlock,
    ImageBlock,
    TextBlock,
)
from llama_index.core.llms import LLM, ChatMessage, MessageRole
from pydantic import Field

from private_gpt.components.chat.processors.chat_history.multimodality.audio_preprocessor import (
    preprocess_audio_message,
)
from private_gpt.components.chat.processors.chat_history.multimodality.image_preprocessor import (
    preprocess_image_message,
)
from private_gpt.components.chat.processors.chat_history.multimodality.multimodality_preprocessor import (
    preprocess_multimodal_history,
    preprocess_multimodal_message,
)
from private_gpt.components.chat.processors.chat_history.multimodality.utils import (
    extract_audio_blocks,
    extract_image_blocks,
    requires_image_preprocessing,
)
from private_gpt.components.llm.custom.mock import FunctionCallingLLMMock
from private_gpt.components.multimodality.image_handler import (
    ExtractionContent,
    ExtractionEvaluation,
    ExtractionStrategy,
)

if TYPE_CHECKING:
    from private_gpt.components.chat.processors.chat_history.multimodality.models import (
        MultimodalProcessingResponse,
    )


class MockLLM(FunctionCallingLLMMock):
    responses: list[Any] = Field(default_factory=list)
    call_count: int = Field(default=0)
    messages_history: list[list[ChatMessage]] = Field(default_factory=list)

    def __init__(self, responses: list[Any] | None = None, **kwargs: Any) -> None:
        # Initialize parent class with only its expected parameters
        super().__init__(**kwargs)

        # Set our custom fields after parent initialization
        if responses is not None:
            self.responses = responses
        self.responses = self.responses or []
        self.call_count = 0
        self.messages_history = []

    async def astructured_chat(
        self, output_cls: type, messages: list[ChatMessage], **kwargs: Any
    ) -> Any:
        self.messages_history.append(messages)
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        raise IndexError("No more mock responses available")


@pytest.fixture
def main_llm() -> LLM:
    return MockLLM()


@pytest.fixture
def image_llm() -> LLM:
    mock_responses = [
        # Strategy inference
        ExtractionStrategy(
            type="table",
            confidence=0.95,
            language="en",
            has_structure=True,
            increase_contrast=False,
        ),
        # Content extraction (complete)
        ExtractionContent(
            markdown="Describe these images",
            is_complete=True,
        ),
        # Evaluation (passes)
        ExtractionEvaluation(
            score=0.9,
            issues_found=[],
        ),
    ]

    llm = MockLLM(responses=mock_responses)
    return llm


@pytest.fixture
def audio_llm() -> LLM:
    from private_gpt.components.multimodality.audio_handler import (
        TimestampSegment,
        TranscriptionContent,
        TranscriptionEvaluation,
        TranscriptionStrategy,
    )

    mock_responses = [
        TranscriptionStrategy(
            type="speech",
            confidence=0.9,
            language="en",
            has_multiple_speakers=False,
            has_background_noise=False,
            enhance_audio=False,
            speaker_diarization=False,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="Hello, how are you today?",
                )
            ],
            is_complete=True,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="This is chunk two.",
                )
            ],
            is_complete=True,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="This is chunk three.",
                )
            ],
            is_complete=True,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="This is chunk four.",
                )
            ],
            is_complete=True,
        ),
        TranscriptionContent(
            timestamps=[
                TimestampSegment(
                    start=0.0,
                    end=2.5,
                    speaker="Speaker 1",
                    text="This is chunk five.",
                )
            ],
            is_complete=True,
        ),
        TranscriptionEvaluation(
            score=0.9,
            issues_found=[],
            clarity=0.85,
        ),
    ]

    llm = MockLLM(responses=mock_responses)
    return llm


@pytest.fixture
def text_message() -> ChatMessage:
    return ChatMessage(
        role=MessageRole.USER,
        blocks=[TextBlock(text="Hello, how are you?")],
    )


@pytest.fixture
def image_message() -> ChatMessage:
    return ChatMessage(
        role=MessageRole.USER,
        blocks=[
            TextBlock(text="Describe these images"),
            ImageBlock(url="https://picsum.photos/200/300"),
            ImageBlock(url="https://picsum.photos/200/300"),
        ],
    )


@pytest.fixture
def audio_message() -> ChatMessage:
    return ChatMessage(
        role=MessageRole.USER,
        blocks=[
            TextBlock(text="Transcribe this audio"),
            AudioBlock(
                url="https://commondatastorage.googleapis.com/codeskulptor-demos/DDR_assets/Kangaroo_MusiQue_-_The_Neverwritten_Role_Playing_Game.mp3"
            ),
        ],
    )


@pytest.fixture
def multimodal_message() -> ChatMessage:
    return ChatMessage(
        role=MessageRole.USER,
        blocks=[
            TextBlock(text="Process this media"),
            ImageBlock(url="https://picsum.photos/200/300"),
            AudioBlock(
                url="https://commondatastorage.googleapis.com/codeskulptor-demos/DDR_assets/Kangaroo_MusiQue_-_The_Neverwritten_Role_Playing_Game.mp3"
            ),
        ],
    )


class TestImagePreprocessing:
    @pytest.mark.asyncio
    async def test_same_llm_no_preprocessing(
        self, main_llm: LLM, image_message: ChatMessage
    ) -> None:
        responses = []
        async for response in preprocess_image_message(
            main_llm, image_message, main_llm
        ):
            responses.append(response)

        result = responses[-1].message
        assert result is image_message

    @pytest.mark.asyncio
    async def test_different_llm_preprocessing_occurs(
        self, main_llm: LLM, image_llm: LLM, image_message: ChatMessage
    ) -> None:
        responses = []
        async for response in preprocess_image_message(
            main_llm, image_message, image_llm
        ):
            responses.append(response)

        result = responses[-1].message
        assert result is not None
        assert "images in their message" in str(result.blocks[-1].text)
        assert result.role == MessageRole.USER
        assert len(result.blocks) >= 2
        assert isinstance(result.blocks[-1], TextBlock)

    @pytest.mark.asyncio
    async def test_text_message_passes_through(
        self, main_llm: LLM, image_llm: LLM, text_message: ChatMessage
    ) -> None:
        responses = []
        async for response in preprocess_image_message(
            main_llm, text_message, image_llm
        ):
            responses.append(response)

        result = responses[-1].message
        assert result is text_message

    @pytest.mark.asyncio
    async def test_missing_image_llm_raises_error(
        self, main_llm: LLM, image_message: ChatMessage
    ) -> None:
        with pytest.raises(
            ValueError, match="Image blocks found but no image-capable LLM"
        ):
            async for _ in preprocess_image_message(main_llm, image_message, None):
                pass


class TestAudioPreprocessing:
    @pytest.mark.asyncio
    async def test_same_llm_no_preprocessing(
        self, main_llm: LLM, audio_message: ChatMessage
    ) -> None:
        responses = []
        async for response in preprocess_audio_message(
            main_llm, audio_message, main_llm
        ):
            responses.append(response)

        result = responses[-1].message
        assert result is audio_message

    @pytest.mark.asyncio
    async def test_different_llm_preprocessing_occurs(
        self, main_llm: LLM, audio_llm: LLM, audio_message: ChatMessage
    ) -> None:
        responses = []
        async for response in preprocess_audio_message(
            main_llm, audio_message, audio_llm
        ):
            responses.append(response)

        result = responses[-1].message
        assert result is not None
        assert "audios in their message" in str(result.blocks[-1].text)
        assert result.role == MessageRole.USER
        assert len(result.blocks) >= 2
        assert isinstance(result.blocks[-1], TextBlock)

    @pytest.mark.asyncio
    async def test_text_message_passes_through(
        self, main_llm: LLM, audio_llm: LLM, text_message: ChatMessage
    ) -> None:
        responses = []
        async for response in preprocess_audio_message(
            main_llm, text_message, audio_llm
        ):
            responses.append(response)

        result = responses[-1].message
        assert result is text_message

    @pytest.mark.asyncio
    async def test_missing_audio_llm_raises_error(
        self, main_llm: LLM, audio_message: ChatMessage
    ) -> None:
        with pytest.raises(
            ValueError, match="Audio blocks found but no audio-capable LLM"
        ):
            async for _ in preprocess_audio_message(main_llm, audio_message, None):
                pass


class TestMultimodalPreprocessing:
    @pytest.mark.asyncio
    async def test_same_llms_no_preprocessing(
        self, main_llm: LLM, multimodal_message: ChatMessage
    ) -> None:
        responses: list[MultimodalProcessingResponse] = []
        async for response in preprocess_multimodal_message(
            main_llm,
            multimodal_message.model_copy(),
            image_multimodal_llm=main_llm,
            audio_multimodal_llm=main_llm,
        ):
            responses.append(response)

        result = responses[-1].modified_message
        assert result.content == multimodal_message.content
        assert result.role == multimodal_message.role

    @pytest.mark.asyncio
    async def test_image_only_preprocessing(
        self, main_llm: LLM, image_llm: LLM, image_message: ChatMessage
    ) -> None:
        responses: list[MultimodalProcessingResponse] = []
        async for response in preprocess_multimodal_message(
            main_llm,
            image_message,
            image_multimodal_llm=image_llm,
            audio_multimodal_llm=main_llm,
        ):
            responses.append(response)

        result = responses[-1].modified_message
        assert len(result.blocks) >= 2
        assert isinstance(result.blocks[-1], TextBlock)
        assert "images in their message" in result.blocks[-1].text

    @pytest.mark.asyncio
    async def test_audio_only_preprocessing(
        self,
        main_llm: LLM,
        image_llm: LLM,
        audio_llm: LLM,
        multimodal_message: ChatMessage,
    ) -> None:
        # Audio processing now yields failed status instead of raising directly
        responses: list[MultimodalProcessingResponse] = []
        async for response in preprocess_multimodal_message(
            main_llm,
            multimodal_message.model_copy(),
            image_multimodal_llm=image_llm,
            audio_multimodal_llm=audio_llm,
        ):
            responses.append(response)

        result = responses[-1].modified_message
        assert len(result.blocks) >= 2
        assert isinstance(result.blocks[-1], TextBlock)
        assert "audios in their message" in result.blocks[-1].text

    @pytest.mark.asyncio
    async def test_text_message_passes_through(
        self,
        main_llm: LLM,
        image_llm: LLM,
        audio_llm: LLM,
        text_message: ChatMessage,
    ) -> None:
        responses: list[MultimodalProcessingResponse] = []
        async for response in preprocess_multimodal_message(
            main_llm,
            text_message,
            image_multimodal_llm=image_llm,
            audio_multimodal_llm=audio_llm,
        ):
            responses.append(response)

        result = responses[-1].modified_message
        assert result.content == text_message.content
        assert result.role == text_message.role


class TestHistoryPreprocessing:
    @pytest.mark.asyncio
    async def test_same_llm_no_preprocessing(
        self, main_llm: LLM, text_message: ChatMessage, image_message: ChatMessage
    ) -> None:
        history = [text_message, image_message]

        responses: list[MultimodalProcessingResponse] = []
        async for response in preprocess_multimodal_history(
            main_llm,
            history,
            image_multimodal_llm=main_llm,
            audio_multimodal_llm=main_llm,
        ):
            responses.append(response)

        result = responses[-1].chat_history
        assert len(result) == len(history)
        assert result[0].content == text_message.content
        assert result[1].content == image_message.content

    @pytest.mark.asyncio
    async def test_processes_most_recent_user_message(
        self,
        main_llm: LLM,
        image_llm: LLM,
        text_message: ChatMessage,
        image_message: ChatMessage,
    ) -> None:
        history = [
            ChatMessage(
                role=MessageRole.USER,
                content="First",
                blocks=[ImageBlock(url="https://picsum.photos/300/300")],
            ),
            ChatMessage(role=MessageRole.ASSISTANT, content="Response", blocks=[]),
            image_message,
        ]

        responses: list[MultimodalProcessingResponse] = []
        async for response in preprocess_multimodal_history(
            main_llm, history, image_multimodal_llm=image_llm, audio_multimodal_llm=None
        ):
            responses.append(response)

        result = responses[-1].chat_history
        assert len(result) == 3
        assert len(result[2].blocks) >= 2
        assert isinstance(result[2].blocks[-1], TextBlock)
        assert "images in their message" in result[2].blocks[-1].text

        assert result[0].content == "First"
        assert all(isinstance(block, TextBlock) for block in result[0].blocks)

    @pytest.mark.asyncio
    async def test_empty_history_returns_none(
        self, main_llm: LLM, image_llm: LLM
    ) -> None:
        responses: list[MultimodalProcessingResponse] = []
        async for response in preprocess_multimodal_history(
            main_llm, None, image_multimodal_llm=image_llm, audio_multimodal_llm=None
        ):
            responses.append(response)

        assert len(responses) == 1
        assert responses[0].chat_history is None

        responses: list[MultimodalProcessingResponse] = []
        async for response in preprocess_multimodal_history(
            main_llm, [], image_multimodal_llm=image_llm, audio_multimodal_llm=None
        ):
            responses.append(response)

        assert len(responses) == 1
        assert responses[0].chat_history == []


class TestTypeChecking:
    def test_llm_objects_identity_comparison(
        self, main_llm: LLM, image_llm: LLM
    ) -> None:
        assert not requires_image_preprocessing(main_llm, main_llm)
        assert requires_image_preprocessing(main_llm, image_llm)
        assert requires_image_preprocessing(main_llm, None)

    def test_extract_functions_return_correct_types(
        self, multimodal_message: ChatMessage
    ) -> None:
        images = extract_image_blocks(multimodal_message)
        audios = extract_audio_blocks(multimodal_message)

        assert isinstance(images, list)
        assert isinstance(audios, list)
        assert len(images) == 1
        assert len(audios) == 1
        assert isinstance(images[0], ImageBlock)
        assert isinstance(audios[0], AudioBlock)


@pytest.mark.parametrize(
    ("same_image_llm", "same_audio_llm", "expected_preprocessing"),
    [
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (False, False, True),
    ],
)
class TestParametrizedPreprocessing:
    @pytest.mark.asyncio
    async def test_preprocessing_decision_logic(
        self,
        main_llm: LLM,
        image_llm: LLM,
        audio_llm: LLM,
        multimodal_message: ChatMessage,
        same_image_llm: bool,
        same_audio_llm: bool,
        expected_preprocessing: bool,
    ) -> None:
        actual_image_llm = main_llm if same_image_llm else image_llm
        actual_audio_llm = main_llm if same_audio_llm else audio_llm

        responses: list[MultimodalProcessingResponse] = []
        async for response in preprocess_multimodal_message(
            main_llm,
            multimodal_message.model_copy(),
            image_multimodal_llm=actual_image_llm,
            audio_multimodal_llm=actual_audio_llm,
        ):
            responses.append(response)

        result = responses[-1].modified_message
        if expected_preprocessing:
            assert result.content != multimodal_message.content
            if not same_image_llm:
                assert len(result.blocks) >= 2
                assert any(
                    "images in their message" in block.text
                    for block in result.blocks
                    if isinstance(block, TextBlock)
                )
            if not same_audio_llm:
                assert any(
                    "audios in their message" in block.text
                    for block in result.blocks
                    if isinstance(block, TextBlock)
                )
        else:
            assert result.content == multimodal_message.content
            assert result.role == multimodal_message.role
