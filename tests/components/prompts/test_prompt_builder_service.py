from unittest.mock import Mock

import pytest
from llama_index.core import PromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore, TextNode

from private_gpt.components.prompts.prompt_builder import PromptBuilderService
from private_gpt.di import get_global_injector


@pytest.fixture
def prompt_builder() -> PromptBuilderService:
    return get_global_injector().get(PromptBuilderService)


@pytest.fixture
def sample_nodes() -> list[NodeWithScore]:
    return [
        NodeWithScore(
            node=TextNode(
                text="Solar panels convert sunlight to electricity using photovoltaic cells.",
                metadata={"file_name": "energy.txt", "page_label": "1"},
            ),
            score=0.9,
        ),
        NodeWithScore(
            node=TextNode(
                text="Wind turbines generate electricity by using wind to rotate blades.",
                metadata={"file_name": "energy.txt", "page_label": "2"},
            ),
            score=0.8,
        ),
    ]


@pytest.fixture
def empty_node_list() -> list[NodeWithScore]:
    return []


@pytest.mark.parametrize(
    ("question", "chat_history", "max_words", "few_shots"),
    [
        (
            "What is their efficiency?",
            "User: Tell me about solar panels\nAI: Solar panels convert sunlight into electricity using photovoltaic cells.",
            25,
            True,
        ),
        (
            "What are the main use cases?",
            "User: Explain Python generators\nAI: Python generators are functions that can pause and resume their execution state.",
            15,
            False,
        ),
        (
            "How many types are there?",
            "User: Tell me about design patterns\nAI: Design patterns are reusable solutions to common software design problems.",
            10,
            True,
        ),
        (
            "What is the capital of France?",
            "User: Can you tell me about the UK?\nAI: The UK consists of England, Scotland, Wales, and Northern Ireland.",
            20,
            False,
        ),
        # Edge case: Empty chat history
        (
            "What is the best programming language?",
            "",
            20,
            True,
        ),
        # Edge case: Very large max_words
        (
            "What are neural networks?",
            "User: Tell me about AI\nAI: AI is a broad field of computer science.",
            1000,
            False,
        ),
        # Edge case: Null max_words
        (
            "What is the best programming language?",
            "User: Can you tell me about the UK?\nAI: The UK consists of England, Scotland, Wales, and Northern Ireland.",
            None,
            True,
        ),
    ],
)
def test_create_chat_condense_prompt(
    prompt_builder: PromptBuilderService,
    question: str,
    chat_history: str,
    max_words: int | None,
    few_shots: bool,
) -> None:
    prompt = prompt_builder.create_chat_condense_prompt(
        question=question,
        chat_history=chat_history,
        max_words=max_words,
        few_shots=few_shots,
    )

    formatted = prompt.format()

    assert (
        "rewrite follow-up questions into clear, standalone questions"
        in formatted.lower()
    )
    assert question in formatted

    if max_words:
        assert str(max_words) in formatted
    else:
        assert "less than" not in formatted

    if chat_history:
        assert chat_history in formatted

    if few_shots:
        assert "Examples:" in formatted
    else:
        assert "Examples:" not in formatted


def test_create_context_prompt_with_nodes(
    prompt_builder: PromptBuilderService, sample_nodes: list[NodeWithScore]
) -> None:
    prompt, _ = prompt_builder.create_context_prompt(
        nodes=sample_nodes, included_in_system_prompt=True
    )
    formatted = prompt.format()

    assert "Context Information" in formatted
    assert "Solar panels convert sunlight to electricity" in formatted
    assert "Wind turbines generate electricity" in formatted


def test_create_context_prompt_with_token_limit(
    prompt_builder: PromptBuilderService, sample_nodes: list[NodeWithScore]
) -> None:
    # Simple tokenizer function for testing
    def simple_tokenizer(text: str) -> list[str]:
        return text.split()

    # Test with small token limit
    prompt, _ = prompt_builder.create_context_prompt(
        nodes=sample_nodes,
        token_limit=10,
        tokenizer_fn=simple_tokenizer,
        included_in_system_prompt=True,
    )

    formatted = prompt.format()
    assert "Context Information" in formatted
    tokens = simple_tokenizer(formatted)
    assert len(tokens) < 19  # Reasonable limit for truncated content


def test_create_context_prompt_empty(prompt_builder: PromptBuilderService) -> None:
    prompt, _ = prompt_builder.create_context_prompt(nodes=None)
    formatted = prompt.format()
    assert formatted == ""


def test_create_context_prompt_empty_list(
    prompt_builder: PromptBuilderService, empty_node_list: list[NodeWithScore]
) -> None:
    prompt, _ = prompt_builder.create_context_prompt(nodes=empty_node_list)
    formatted = prompt.format()
    assert formatted == ""


def test_create_citation_prompt(
    prompt_builder: PromptBuilderService, sample_nodes: list[NodeWithScore]
) -> None:
    prompt = prompt_builder.create_citation_prompt(nodes=sample_nodes)
    formatted = prompt.format()
    assert "citation" in formatted.lower()
    assert "citation_protocol" in formatted.lower()


def test_create_citation_prompt_empty(prompt_builder: PromptBuilderService) -> None:
    prompt = prompt_builder.create_citation_prompt(nodes=None)
    formatted = prompt.format()
    assert formatted == ""


def test_create_citation_prompt_empty_list(
    prompt_builder: PromptBuilderService, empty_node_list: list[NodeWithScore]
) -> None:
    prompt = prompt_builder.create_citation_prompt(nodes=empty_node_list)
    formatted = prompt.format()
    assert formatted == ""


def test_create_citation_prompt_no_metadata(
    prompt_builder: PromptBuilderService,
) -> None:
    # Create nodes with no metadata
    no_metadata_nodes = [
        NodeWithScore(
            node=TextNode(
                text="Text without metadata",
                extra_info={},  # Empty metadata
            ),
            score=0.9,
        ),
    ]

    prompt = prompt_builder.create_citation_prompt(nodes=no_metadata_nodes)
    formatted = prompt.format()

    # Should still produce a valid citation prompt
    assert "citation" in formatted.lower()


@pytest.mark.parametrize(
    ("system_prompt", "user_query", "additional_instructions"),
    [
        (
            "You are a helpful assistant.",
            "Explain quantum computing",
            "Keep it simple and focus on practical applications",
        ),
        (None, "History of the internet", None),
        (None, "", "Make it comprehensive"),  # Edge case: Empty query
        (None, "Machine learning basics", ""),  # Edge case: Empty instructions
    ],
)
def test_create_summary_prompt(
    prompt_builder: PromptBuilderService,
    system_prompt: str | None,
    user_query: str,
    additional_instructions: str | None,
) -> None:
    prompt = prompt_builder.create_summary_prompt(
        system_prompt=system_prompt,
        user_query=user_query,
        additional_instructions=additional_instructions,
    )
    formatted = prompt.format()

    if system_prompt:
        assert system_prompt in formatted

    # Verify the template structure
    assert user_query in formatted
    assert "Rules:" in formatted.lower() or "rules" in formatted.lower()

    # Check conditional rendering of additional instructions
    if additional_instructions:
        assert additional_instructions in formatted
    else:
        assert "Additional Instructions:" not in formatted
        assert "additional instructions:" not in formatted.lower()


def test_create_summary_prompt_no_instructions(
    prompt_builder: PromptBuilderService,
) -> None:
    user_query = "Explain quantum computing"

    prompt = prompt_builder.create_summary_prompt(user_query=user_query)
    formatted = prompt.format()

    assert user_query in formatted
    assert "Rules:" in formatted.lower() or "rules" in formatted.lower()
    assert "Additional Instructions:" not in formatted
    assert "additional instructions:" not in formatted.lower()


@pytest.fixture
def sample_chat_history() -> list[ChatMessage]:
    return [
        ChatMessage(role=MessageRole.USER, content="What is machine learning?"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Machine learning is a subset of AI that enables computers to learn from data.",
        ),
        ChatMessage(role=MessageRole.USER, content="Can you give me an example?"),
        ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Sure! Email spam detection uses ML to classify emails as spam or legitimate.",
        ),
    ]


@pytest.fixture
def empty_chat_history() -> list[ChatMessage]:
    return []


@pytest.fixture
def mock_messages_to_history_str():
    def mock_fn(messages):
        return "\n".join([f"{msg.role.value}: {msg.content}" for msg in messages])

    return mock_fn


@pytest.mark.parametrize(
    (
        "user_query",
        "chat_history_fixture",
        "system_prompt",
        "max_words",
        "few_shots",
        "use_custom_fn",
    ),
    [
        (
            "What are the main applications?",
            "sample_chat_history",
            "You are a helpful AI assistant.",
            100,
            True,
            False,
        ),
        (
            "Explain this concept further",
            "sample_chat_history",
            None,
            50,
            False,
            True,
        ),
        (
            "What is the next step?",
            "empty_chat_history",
            "Be concise and accurate.",
            None,
            True,
            False,
        ),
        (
            "How does this work?",
            None,
            "System prompt test",
            200,
            False,
            False,
        ),
        # Edge case: PromptTemplate as system_prompt
        (
            "Advanced question",
            "sample_chat_history",
            "template_prompt",
            75,
            True,
            True,
        ),
        # Edge case: Zero max_words
        (
            "Brief question",
            "sample_chat_history",
            None,
            0,
            False,
            False,
        ),
        # Edge case: Very large max_words
        (
            "Detailed explanation needed",
            "sample_chat_history",
            "Detailed system prompt",
            10000,
            True,
            False,
        ),
    ],
)
def test_create_summary_history_in_details(
    prompt_builder: PromptBuilderService,
    sample_chat_history: list[ChatMessage],
    empty_chat_history: list[ChatMessage],
    mock_messages_to_history_str,
    user_query: str,
    chat_history_fixture: str | None,
    system_prompt: str | None,
    max_words: int | None,
    few_shots: bool,
    use_custom_fn: bool,
) -> None:
    # Setup chat history based on fixture name
    if chat_history_fixture == "sample_chat_history":
        chat_history = sample_chat_history
    elif chat_history_fixture == "empty_chat_history":
        chat_history = empty_chat_history
    else:
        chat_history = None

    # Handle PromptTemplate case
    if system_prompt == "template_prompt":
        mock_template = Mock(spec=PromptTemplate)
        mock_template.format.return_value = "Formatted template content"
        system_prompt = mock_template

    # Setup custom function
    custom_fn = mock_messages_to_history_str if use_custom_fn else None

    # Test empty chat history returns empty template
    if chat_history_fixture == "empty_chat_history":
        result = prompt_builder.create_summary_history_in_details(
            user_query=user_query,
            chat_history=chat_history,
            system_prompt=system_prompt,
            max_words=max_words,
            few_shots=few_shots,
            messages_to_history_str_fn=custom_fn,
        )
        assert result.template == ""
        return

    # Test normal cases
    result = prompt_builder.create_summary_history_in_details(
        user_query=user_query,
        chat_history=chat_history,
        system_prompt=system_prompt,
        max_words=max_words,
        few_shots=few_shots,
        messages_to_history_str_fn=custom_fn,
    )

    # Verify result is a proper prompt template
    assert hasattr(result, "format")
    formatted = result.format()

    # Verify user query is included
    assert user_query in formatted or "reply to the following user content" in formatted

    # Verify chat history handling
    if chat_history:
        if use_custom_fn:
            # Custom function was used
            assert any(msg.content in formatted for msg in chat_history)
        else:
            # Default function behavior
            assert any(msg.content in formatted for msg in chat_history)

    # Verify max_words parameter
    if max_words and max_words != 0:
        assert str(max_words) in formatted

    # Verify few_shots parameter
    if few_shots:
        assert "Examples" in formatted or "Example" in formatted

    # Verify system prompt handling
    if isinstance(system_prompt, Mock):
        system_prompt.format.assert_called_once()
    elif system_prompt and chat_history_fixture != "empty_chat_history":
        # System prompt should be processed appropriately
        assert result is not None


@pytest.mark.parametrize(
    (
        "chat_history_fixture",
        "system_prompt",
        "max_words",
        "few_shots",
        "use_custom_fn",
    ),
    [
        (
            "sample_chat_history",
            "You are a summarization assistant.",
            150,
            True,
            False,
        ),
        (
            "sample_chat_history",
            None,
            75,
            False,
            True,
        ),
        (
            "empty_chat_history",
            "Brief and accurate",
            None,
            True,
            False,
        ),
        (
            None,
            "System prompt for no history",
            100,
            False,
            False,
        ),
        # Edge case: PromptTemplate as system_prompt
        (
            "sample_chat_history",
            "template_prompt",
            50,
            True,
            True,
        ),
        # Edge case: Zero max_words
        (
            "sample_chat_history",
            None,
            0,
            False,
            False,
        ),
        # Edge case: Very large max_words
        (
            "sample_chat_history",
            "Comprehensive summarization",
            5000,
            True,
            False,
        ),
    ],
)
def test_create_summary_history_approximately(
    prompt_builder: PromptBuilderService,
    sample_chat_history: list[ChatMessage],
    empty_chat_history: list[ChatMessage],
    mock_messages_to_history_str,
    chat_history_fixture: str | None,
    system_prompt: str | None,
    max_words: int | None,
    few_shots: bool,
    use_custom_fn: bool,
) -> None:
    # Setup chat history based on fixture name
    if chat_history_fixture == "sample_chat_history":
        chat_history = sample_chat_history
    elif chat_history_fixture == "empty_chat_history":
        chat_history = empty_chat_history
    else:
        chat_history = None

    # Handle PromptTemplate case
    if system_prompt == "template_prompt":
        mock_template = Mock(spec=PromptTemplate)
        mock_template.format.return_value = (
            "Formatted template content for approximation"
        )
        system_prompt = mock_template

    # Setup custom function
    custom_fn = mock_messages_to_history_str if use_custom_fn else None

    # Test empty chat history returns empty template
    if chat_history_fixture == "empty_chat_history":
        result = prompt_builder.create_summary_history_approximately(
            chat_history=chat_history,
            system_prompt=system_prompt,
            max_words=max_words,
            few_shots=few_shots,
            messages_to_history_str_fn=custom_fn,
        )
        assert result.template == ""
        return

    # Test normal cases
    result = prompt_builder.create_summary_history_approximately(
        chat_history=chat_history,
        system_prompt=system_prompt,
        max_words=max_words,
        few_shots=few_shots,
        messages_to_history_str_fn=custom_fn,
    )

    # Verify result is a proper prompt template
    assert hasattr(result, "format")
    formatted = result.format()

    # Verify summarization instruction is present
    assert "summarize" in formatted.lower() or "json array" in formatted.lower()

    # Verify chat history handling
    if chat_history:
        if use_custom_fn:
            # Custom function was used
            assert any(msg.content in formatted for msg in chat_history)
        else:
            # Default function behavior
            assert any(msg.content in formatted for msg in chat_history)

    # Verify max_words parameter
    if max_words and max_words != 0:
        assert str(max_words) in formatted

    # Verify few_shots parameter
    if few_shots:
        assert "Examples" in formatted or "Example" in formatted

    # Verify JSON format requirement
    assert "json" in formatted.lower()
    assert "array" in formatted.lower()

    # Verify role specification
    assert "user" in formatted
    assert "assistant" in formatted

    # Verify system prompt handling
    if isinstance(system_prompt, Mock):
        system_prompt.format.assert_called_once()
    elif system_prompt and chat_history_fixture != "empty_chat_history":
        # System prompt should be processed appropriately
        assert result is not None


# ---------------------------------------------------------------------------
# Tests for new create_* guidelines methods
# ---------------------------------------------------------------------------


def test_create_thinking_guidelines(prompt_builder: PromptBuilderService) -> None:
    prompt = prompt_builder.create_thinking_guidelines()
    formatted = prompt.format()
    assert "thinking" in formatted.lower()
    assert formatted != ""


def test_create_thinking_guidelines_few_shots(
    prompt_builder: PromptBuilderService,
) -> None:
    with_shots = prompt_builder.create_thinking_guidelines(few_shots=True)
    without_shots = prompt_builder.create_thinking_guidelines(few_shots=False)
    assert "Examples" in with_shots.format() or "Good thinking" in with_shots.format()
    assert len(with_shots.format()) > len(without_shots.format())


def test_create_citation_guidelines(prompt_builder: PromptBuilderService) -> None:
    prompt = prompt_builder.create_citation_guidelines()
    formatted = prompt.format()
    assert "citation" in formatted.lower()
    assert formatted != ""


def test_create_tool_instructions_unknown_tool_returns_empty(
    prompt_builder: PromptBuilderService,
) -> None:
    from private_gpt.components.prompts.prompt_builder import _ToolNamespace

    prompt = prompt_builder.create_tool_instructions(
        "nonexistent_tool_xyz", _ToolNamespace({})
    )
    assert prompt.format() == ""


def test_create_tool_instructions_known_tool(
    prompt_builder: PromptBuilderService,
) -> None:
    from private_gpt.components.chat.models.chat_config_models import ToolSpec
    from private_gpt.components.prompts.prompt_builder import _build_tool_namespace

    tool = ToolSpec(name="web_search")
    namespace = _build_tool_namespace([tool])
    prompt = prompt_builder.create_tool_instructions("web_search", namespace)
    assert prompt.format() != ""


def test_seed_tool_instructions_skips_explicit(
    prompt_builder: PromptBuilderService,
) -> None:
    from private_gpt.components.chat.models.chat_config_models import ToolSpec

    tool = ToolSpec(name="web_search", instructions="Custom override")
    seeded = prompt_builder.seed_tool_instructions([tool])
    assert seeded[0].instructions == "Custom override"


def test_seed_tool_instructions_empty_string_suppresses(
    prompt_builder: PromptBuilderService,
) -> None:
    from private_gpt.components.chat.models.chat_config_models import ToolSpec

    tool = ToolSpec(name="web_search", instructions="")
    seeded = prompt_builder.seed_tool_instructions([tool])
    assert seeded[0].instructions == ""
