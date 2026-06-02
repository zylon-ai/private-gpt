import base64
import builtins
from typing import Any

import pytest
from llama_index.core.base.llms.types import ImageBlock
from llama_index.core.llms import ChatMessage
from pydantic import Field

from private_gpt.components.llm.custom.mock import FunctionCallingLLMMock
from private_gpt.components.multimodality.image_handler import (
    ExtractionContent,
    ExtractionEvaluation,
    ExtractionStrategy,
    ImageProcessingWorkflow,
)


class MockFlexibleModel:
    def __init__(self, **kwargs: Any) -> None:
        self._data = kwargs

    def dict(self) -> dict[str, Any]:
        return self._data

    def model_dump(self) -> builtins.dict[str, Any]:
        return self._data

    def __getattr__(self, item: str) -> Any:
        return self._data.get(item)


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


def create_test_image_block() -> ImageBlock:
    # Create a minimal base64 encoded image (1x1 PNG)
    image_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    return ImageBlock(image=image_data, image_mimetype="image/png")


@pytest.fixture
def test_image_blocks() -> list[ImageBlock]:
    return [create_test_image_block()]


@pytest.mark.asyncio
async def test_clean_forward_flow(test_image_blocks: list[ImageBlock]) -> None:
    """Test 1: Clean forward flow without preprocessing or evaluation."""
    mock_responses = [
        # Strategy inference response
        ExtractionStrategy(
            type="text",
            confidence=0.9,
            language="en",
            has_structure=False,
            increase_contrast=False,
        ),
        # Content extraction response (complete)
        ExtractionContent(
            markdown="# Test Content\nThis is extracted text.", is_complete=True
        ),
    ]

    mock_llm = MockLLM(responses=mock_responses)
    workflow = ImageProcessingWorkflow(image_multimodal_llm=mock_llm)

    result = await workflow.run(
        image_blocks=test_image_blocks,
        user_query="Extract text from image",
        max_iterations=3,
        enable_preprocessing=False,
        enable_evaluation=False,
        kwargs={},
    )

    assert result is not None
    assert "Test Content" in result.description
    assert result.evaluation is None
    assert mock_llm.call_count == 2  # Strategy + content extraction


@pytest.mark.asyncio
async def test_incomplete_content_with_retries(
    test_image_blocks: list[ImageBlock],
) -> None:
    """Test 2: Evaluation=False, model generates incomplete content and retries."""
    mock_responses = [
        # Strategy inference
        ExtractionStrategy(
            type="text",
            confidence=0.8,
            language="en",
            has_structure=True,
            increase_contrast=True,
        ),
        # First content extraction (incomplete, using FlexibleModel)
        MockFlexibleModel(
            markdown="# Partial Content\nThis is incomplete...", is_complete=False
        ),
        # Second content extraction (complete)
        ExtractionContent(
            markdown="# Complete Content\nThis is the complete extracted text.",
            is_complete=True,
        ),
    ]

    mock_llm = MockLLM(responses=mock_responses)
    workflow = ImageProcessingWorkflow(image_multimodal_llm=mock_llm)

    result = await workflow.run(
        image_blocks=test_image_blocks,
        user_query="Extract structured content",
        max_iterations=3,
        enable_preprocessing=True,
        enable_evaluation=False,
        kwargs={},
    )

    assert result is not None
    assert "Partial Content" in result.description
    assert "Complete Content" in result.description
    assert result.evaluation is None
    assert mock_llm.call_count == 3  # Strategy + 2 content extractions


@pytest.mark.asyncio
async def test_evaluation_passes(test_image_blocks: list[ImageBlock]) -> None:
    """Test 3: Evaluation=True, content is complete and passes evaluation."""
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
            markdown="| Column 1 | Column 2 |\n|----------|----------|\n| Data 1   | Data 2   |",
            is_complete=True,
        ),
        # Evaluation (passes)
        ExtractionEvaluation(
            score=0.9,
            issues_found=[],
        ),
    ]

    mock_llm = MockLLM(responses=mock_responses)
    workflow = ImageProcessingWorkflow(image_multimodal_llm=mock_llm)

    result = await workflow.run(
        image_blocks=test_image_blocks,
        user_query="Extract table data",
        max_iterations=3,
        enable_preprocessing=True,
        enable_evaluation=True,
        kwargs={},
    )

    assert result is not None
    assert "| Column 1 | Column 2 |" in result.description
    assert result.evaluation is not None
    assert result.evaluation.score == 0.9
    assert len(result.evaluation.issues_found) == 0
    assert mock_llm.call_count == 3  # Strategy + content + evaluation


@pytest.mark.asyncio
async def test_evaluation_fails_triggers_retry(
    test_image_blocks: list[ImageBlock],
) -> None:
    """Test 4: Evaluation fails, triggers retry with errors/suggestions."""
    mock_responses = [
        # Strategy inference
        ExtractionStrategy(
            type="form",
            confidence=0.7,
            language="en",
            has_structure=True,
            increase_contrast=True,
        ),
        # First content extraction (complete but poor quality)
        ExtractionContent(markdown="Some incomplete form data", is_complete=True),
        # First evaluation (fails)
        ExtractionEvaluation(
            score=0.5,  # Below threshold of 0.7
            issues_found=["Missing field labels", "Incomplete data"],
        ),
        # Second content extraction (after retry with suggestions)
        ExtractionContent(
            markdown="# Complete Form\n**Name:** John Doe\n**Email:** john@example.com",
            is_complete=True,
        ),
        # Second evaluation (passes)
        ExtractionEvaluation(
            score=0.85,
            issues_found=[],
        ),
    ]

    mock_llm = MockLLM(responses=mock_responses)
    workflow = ImageProcessingWorkflow(image_multimodal_llm=mock_llm)

    result = await workflow.run(
        image_blocks=test_image_blocks,
        user_query="Extract form data",
        max_iterations=3,
        enable_preprocessing=True,
        enable_evaluation=True,
        kwargs={},
    )

    assert result is not None
    assert "Some incomplete form data" not in result.description
    assert "Complete Form" in result.description
    assert result.evaluation is not None
    assert result.evaluation.score == 0.85
    assert len(result.evaluation.issues_found) == 0
    assert mock_llm.call_count == 5  # Strategy + content + eval + content + eval


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enable_preprocessing", "enable_evaluation", "expected_calls"),
    [
        (False, False, 2),  # Strategy + content only
        (True, False, 2),  # Strategy + content (no preprocessing needed)
        (False, True, 3),  # Strategy + content + evaluation
        (True, True, 3),  # Strategy + content + evaluation (no preprocessing needed)
    ],
)
async def test_configuration_options(
    test_image_blocks: list[ImageBlock],
    enable_preprocessing: bool,
    enable_evaluation: bool,
    expected_calls: int,
) -> None:
    """Test different configuration combinations."""
    mock_responses = [
        # Strategy (no contrast enhancement needed)
        ExtractionStrategy(
            type="text",
            confidence=0.8,
            language="en",
            has_structure=False,
            increase_contrast=False,
        ),
        # Content extraction
        ExtractionContent(markdown="Test content", is_complete=True),
        # Evaluation (good quality)
        ExtractionEvaluation(
            score=0.9,
        ),
    ]

    mock_llm = MockLLM(responses=mock_responses)
    workflow = ImageProcessingWorkflow(image_multimodal_llm=mock_llm)

    result = await workflow.run(
        image_blocks=test_image_blocks,
        user_query="Test query",
        max_iterations=2,
        enable_preprocessing=enable_preprocessing,
        enable_evaluation=enable_evaluation,
        kwargs={},
    )

    assert result is not None
    assert mock_llm.call_count == expected_calls


@pytest.mark.asyncio
async def test_max_iterations_reached(test_image_blocks: list[ImageBlock]) -> None:
    """Test 5: Max iterations reached without complete content."""
    mock_responses = [
        ExtractionStrategy(
            type="text",
            confidence=0.8,
            language="en",
            has_structure=False,
            increase_contrast=False,
        ),
        ExtractionContent(markdown="Partial content 1", is_complete=False),
        ExtractionContent(markdown="Partial content 2", is_complete=False),
    ]

    mock_llm = MockLLM(responses=mock_responses)
    workflow = ImageProcessingWorkflow(
        image_multimodal_llm=mock_llm,
    )

    result = await workflow.run(
        image_blocks=test_image_blocks,
        user_query="Extract text",
        max_iterations=2,
        enable_preprocessing=False,
        enable_evaluation=False,
        kwargs={},
    )

    assert result is not None
    assert "Partial content 1" in result.description
    assert "Partial content 2" in result.description
    assert mock_llm.call_count == 3


# TODO: Re-enable when we merged with images branch
# @pytest.mark.asyncio
# async def test_error_handling_in_strategy_inference(
#     test_image_blocks: list[ImageBlock],
# ) -> None:
#     """Test 6: Error handling during strategy inference."""
#     mock_llm = MagicMock(spec=LLM)
#     mock_llm.astructured_chat = AsyncMock(
#         side_effect=RuntimeError("Strategy inference failed")
#     )
#
#     workflow = ImageProcessingWorkflow(
#         image_multimodal_llm=mock_llm,
#     )
#
#     with pytest.raises(WorkflowRuntimeError):
#         await workflow.run(
#             image_blocks=test_image_blocks,
#             user_query="Test error handling",
#             max_iterations=1,
#             enable_preprocessing=False,
#             enable_evaluation=False,
#             kwargs={},
#         )
