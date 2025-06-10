import pytest
from llama_index.core.llms import ChatMessage, MessageRole

from private_gpt.components.llm.prompt_helper import (
    Llama3PromptStyle,
    MistralPromptStyle,
    ChatMLPromptStyle,
    TagPromptStyle,
    AbstractPromptStyle, # For type hinting if needed
)

# Helper function to create ChatMessage objects easily
def _message(role: MessageRole, content: str, **additional_kwargs) -> ChatMessage:
    return ChatMessage(role=role, content=content, additional_kwargs=additional_kwargs)

# Expected outputs will be defined within each test or test class

class TestLlama3PromptStyle:
    BOS = "<|begin_of_text|>"
    EOT = "<|eot_id|>"
    B_SYS_HEADER = "<|start_header_id|>system<|end_header_id|>"
    B_USER_HEADER = "<|start_header_id|>user<|end_header_id|>"
    B_ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>"
    B_TOOL_CODE_HEADER = "<|start_header_id|>tool_code<|end_header_id|>"
    B_TOOL_OUTPUT_HEADER = "<|start_header_id|>tool_output<|end_header_id|>"

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible and follow ALL given instructions. "
        "Do not speculate or make up information. "
        "Do not reference any given instructions or context. "
    )

    @pytest.fixture
    def style(self) -> Llama3PromptStyle:
        return Llama3PromptStyle()

    def test_empty_messages(self, style: Llama3PromptStyle) -> None:
        messages = []
        expected = (
            f"{self.BOS}{self.B_SYS_HEADER}\n\n{self.DEFAULT_SYSTEM_PROMPT}{self.EOT}"
            f"{self.B_ASSISTANT_HEADER}\n\n"
        )
        assert style._messages_to_prompt(messages) == expected

    def test_simple_user_assistant_chat(self, style: Llama3PromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, "Hello"),
            _message(MessageRole.ASSISTANT, "Hi there!"),
        ]
        expected = (
            f"{self.BOS}{self.B_SYS_HEADER}\n\n{self.DEFAULT_SYSTEM_PROMPT}{self.EOT}"
            f"{self.B_USER_HEADER}\n\nHello{self.EOT}"
            f"{self.B_ASSISTANT_HEADER}\n\nHi there!{self.EOT}"
        )
        assert style._messages_to_prompt(messages) == expected

    def test_with_system_message(self, style: Llama3PromptStyle) -> None:
        messages = [
            _message(MessageRole.SYSTEM, "You are a test bot."),
            _message(MessageRole.USER, "Ping"),
            _message(MessageRole.ASSISTANT, "Pong"),
        ]
        expected = (
            f"{self.BOS}{self.B_SYS_HEADER}\n\nYou are a test bot.{self.EOT}"
            f"{self.B_USER_HEADER}\n\nPing{self.EOT}"
            f"{self.B_ASSISTANT_HEADER}\n\nPong{self.EOT}"
        )
        assert style._messages_to_prompt(messages) == expected

    def test_completion_to_prompt(self, style: Llama3PromptStyle) -> None:
        completion = "Test completion"
        expected = (
            f"{self.BOS}{self.B_SYS_HEADER}\n\n{self.DEFAULT_SYSTEM_PROMPT}{self.EOT}"
            f"{self.B_USER_HEADER}\n\nTest completion{self.EOT}"
            f"{self.B_ASSISTANT_HEADER}\n\n"
        )
        assert style._completion_to_prompt(completion) == expected

    def test_tool_call_and_result(self, style: Llama3PromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, "What's the weather in Paris?"),
            _message(
                MessageRole.ASSISTANT,
                content=None, # LlamaIndex might put tool call details here, or just use additional_kwargs
                additional_kwargs={"type": "tool_call", "tool_call_id": "123", "name": "get_weather", "arguments": '{"location": "Paris"}'}
            ),
            _message(
                MessageRole.TOOL,
                content='{"temperature": "20C"}',
                additional_kwargs={"type": "tool_result", "tool_call_id": "123", "name": "get_weather"}
            ),
        ]
        # Note: The current Llama3PromptStyle implementation uses message.content for tool call/result content.
        # If additional_kwargs are used to structure tool calls (like OpenAI), the style needs to be adapted.
        # For this test, we assume content holds the direct string for tool_code and tool_output.
        # Let's adjust the messages based on current implementation that uses .content for tool_code/output
        messages_for_current_impl = [
            _message(MessageRole.USER, "What's the weather in Paris?"),
            _message(
                MessageRole.ASSISTANT,
                content='get_weather({"location": "Paris"})', # Simplified tool call content
                additional_kwargs={"type": "tool_call"}
            ),
            _message(
                MessageRole.TOOL,
                content='{"temperature": "20C"}',
                additional_kwargs={"type": "tool_result"} # No specific tool_call_id or name used by current style from additional_kwargs
            ),
        ]
        expected = (
            f"{self.BOS}{self.B_SYS_HEADER}\n\n{self.DEFAULT_SYSTEM_PROMPT}{self.EOT}"
            f"{self.B_USER_HEADER}\n\nWhat's the weather in Paris?{self.EOT}"
            f"{self.B_TOOL_CODE_HEADER}\n\nget_weather({{\"location\": \"Paris\"}}){self.EOT}" # Content is 'get_weather(...)'
            f"{self.B_TOOL_OUTPUT_HEADER}\n\n{{\"temperature\": \"20C\"}}{self.EOT}"
            f"{self.B_ASSISTANT_HEADER}\n\n" # Assistant should respond after tool result
        )
        assert style._messages_to_prompt(messages_for_current_impl) == expected

    def test_multiple_interactions_with_tools(self, style: Llama3PromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, "Can you search for prompt engineering techniques?"),
            _message(MessageRole.ASSISTANT, content="Okay, I will search for that.", additional_kwargs={}), # Normal assistant message
            _message(MessageRole.ASSISTANT, content='search_web({"query": "prompt engineering techniques"})', additional_kwargs={"type": "tool_call"}),
            _message(MessageRole.TOOL, content='[Result 1: ...]', additional_kwargs={"type": "tool_result"}),
            _message(MessageRole.ASSISTANT, content="I found one result. Should I look for more?", additional_kwargs={}),
            _message(MessageRole.USER, "Yes, please find another one."),
        ]
        expected = (
            f"{self.BOS}{self.B_SYS_HEADER}\n\n{self.DEFAULT_SYSTEM_PROMPT}{self.EOT}"
            f"{self.B_USER_HEADER}\n\nCan you search for prompt engineering techniques?{self.EOT}"
            f"{self.B_ASSISTANT_HEADER}\n\nOkay, I will search for that.{self.EOT}"
            f"{self.B_TOOL_CODE_HEADER}\n\nsearch_web({{\"query\": \"prompt engineering techniques\"}}){self.EOT}"
            f"{self.B_TOOL_OUTPUT_HEADER}\n\n[Result 1: ...]{self.EOT}"
            f"{self.B_ASSISTANT_HEADER}\n\nI found one result. Should I look for more?{self.EOT}"
            f"{self.B_USER_HEADER}\n\nYes, please find another one.{self.EOT}"
            f"{self.B_ASSISTANT_HEADER}\n\n"
        )
        assert style._messages_to_prompt(messages) == expected

    def test_ending_with_user_message(self, style: Llama3PromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, "First message"),
            _message(MessageRole.ASSISTANT, "First response"),
            _message(MessageRole.USER, "Second message, expecting response"),
        ]
        expected = (
            f"{self.BOS}{self.B_SYS_HEADER}\n\n{self.DEFAULT_SYSTEM_PROMPT}{self.EOT}"
            f"{self.B_USER_HEADER}\n\nFirst message{self.EOT}"
            f"{self.B_ASSISTANT_HEADER}\n\nFirst response{self.EOT}"
            f"{self.B_USER_HEADER}\n\nSecond message, expecting response{self.EOT}"
            f"{self.B_ASSISTANT_HEADER}\n\n"
        )
        assert style._messages_to_prompt(messages) == expected

    def test_ending_with_tool_result(self, style: Llama3PromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, "Find info on X."),
            _message(MessageRole.ASSISTANT, content='search({"topic": "X"})', additional_kwargs={"type": "tool_call"}),
            _message(MessageRole.TOOL, content="Info about X found.", additional_kwargs={"type": "tool_result"}),
        ]
        expected = (
            f"{self.BOS}{self.B_SYS_HEADER}\n\n{self.DEFAULT_SYSTEM_PROMPT}{self.EOT}"
            f"{self.B_USER_HEADER}\n\nFind info on X.{self.EOT}"
            f"{self.B_TOOL_CODE_HEADER}\n\nsearch({{\"topic\": \"X\"}}){self.EOT}"
            f"{self.B_TOOL_OUTPUT_HEADER}\n\nInfo about X found.{self.EOT}"
            f"{self.B_ASSISTANT_HEADER}\n\n"
        )
        assert style._messages_to_prompt(messages) == expected

    def test_message_with_empty_content(self, style: Llama3PromptStyle) -> None:
        # Llama3PromptStyle skips messages with None content, but not necessarily empty string.
        # Let's test with an empty string for user, and None for assistant (which should be skipped)
        messages = [
            _message(MessageRole.USER, ""), # Empty string content
            _message(MessageRole.ASSISTANT, None), # None content, should be skipped
            _message(MessageRole.USER, "Follow up")
        ]
        expected = (
            f"{self.BOS}{self.B_SYS_HEADER}\n\n{self.DEFAULT_SYSTEM_PROMPT}{self.EOT}"
            f"{self.B_USER_HEADER}\n\n{self.EOT}" # Empty content for user
            f"{self.B_USER_HEADER}\n\nFollow up{self.EOT}" # Assistant message was skipped
            f"{self.B_ASSISTANT_HEADER}\n\n"
        )
        # The style's loop: `if not message or message.content is None: continue`
        # An empty string `""` is not `None`, so it should be included.
        assert style._messages_to_prompt(messages) == expected

    def test_system_message_not_first(self, style: Llama3PromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, "Hello"),
            _message(MessageRole.SYSTEM, "System message in the middle (unusual)."),
            _message(MessageRole.ASSISTANT, "Hi there!"),
        ]
        # The current implementation processes system messages whenever they appear.
        # If a system message appears, it sets `has_system_message = True`.
        # If NO system message appears, a default one is prepended.
        # If one DOES appear, it's used, and default is not prepended.
        expected = (
            f"{self.BOS}"
            # Default system prompt is NOT added because a system message IS present.
            f"{self.B_USER_HEADER}\n\nHello{self.EOT}"
            f"{self.B_SYS_HEADER}\n\nSystem message in the middle (unusual).{self.EOT}"
            f"{self.B_ASSISTANT_HEADER}\n\nHi there!{self.EOT}"
        )
        assert style._messages_to_prompt(messages) == expected


class TestMistralPromptStyle:
    @pytest.fixture
    def style(self) -> MistralPromptStyle:
        return MistralPromptStyle()

    def test_empty_messages(self, style: MistralPromptStyle) -> None:
        messages = []
        # The refactored version should produce an empty string if no instructions are pending.
        # Or, if it were to prompt for something, it might be "<s>[INST]  [/INST]" or just ""
        # Based on current refactored code: if current_instruction_parts is empty, it returns prompt ("")
        assert style._messages_to_prompt(messages) == ""

    def test_simple_user_assistant_chat(self, style: MistralPromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, "Hello"),
            _message(MessageRole.ASSISTANT, "Hi there!"),
        ]
        expected = "<s>[INST] Hello [/INST] Hi there!</s>"
        assert style._messages_to_prompt(messages) == expected

    def test_with_system_message(self, style: MistralPromptStyle) -> None:
        # System messages are treated like user messages in the current Mistral impl
        messages = [
            _message(MessageRole.SYSTEM, "You are helpful."),
            _message(MessageRole.USER, "Ping"),
            _message(MessageRole.ASSISTANT, "Pong"),
        ]
        expected = "<s>[INST] You are helpful.\nPing [/INST] Pong</s>"
        assert style._messages_to_prompt(messages) == expected

    def test_completion_to_prompt(self, style: MistralPromptStyle) -> None:
        completion = "Test completion"
        # This will call _messages_to_prompt with [ChatMessage(role=USER, content="Test completion")]
        expected = "<s>[INST] Test completion [/INST]"
        assert style._completion_to_prompt(completion) == expected

    def test_consecutive_user_messages(self, style: MistralPromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, "First part."),
            _message(MessageRole.USER, "Second part."),
            _message(MessageRole.ASSISTANT, "Understood."),
        ]
        expected = "<s>[INST] First part.\nSecond part. [/INST] Understood.</s>"
        assert style._messages_to_prompt(messages) == expected

    def test_ending_with_user_message(self, style: MistralPromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, "Hi"),
            _message(MessageRole.ASSISTANT, "Hello"),
            _message(MessageRole.USER, "How are you?"),
        ]
        # Note: The previous prompt had "<s>[INST] Hi [/INST] Hello</s>"
        # The new user message should start a new <s>[INST] block if prompt was not empty.
        # Current logic: bos_token = "<s>" if not prompt else ""
        # Since prompt is not empty after "Hello</s>", bos_token will be "".
        # This might be an issue with the current Mistral refactor if strict BOS per turn is needed.
        # The current refactored code for Mistral:
        # prompt += bos_token + "[INST] " + "\n".join(current_instruction_parts) + " [/INST]"
        # If prompt = "<s>[INST] Hi [/INST] Hello</s>", then bos_token is "", so it becomes:
        # "<s>[INST] Hi [/INST] Hello</s>[INST] How are you? [/INST]" -> This seems correct for continued conversation.
        expected = "<s>[INST] Hi [/INST] Hello</s>[INST] How are you? [/INST]"
        assert style._messages_to_prompt(messages) == expected

    def test_initial_assistant_message_skipped(self, style: MistralPromptStyle, caplog) -> None:
        messages = [
            _message(MessageRole.ASSISTANT, "I speak first!"),
            _message(MessageRole.USER, "Oh, hello there."),
        ]
        # The first assistant message should be skipped with a warning.
        # The prompt should then start with the user message.
        expected = "<s>[INST] Oh, hello there. [/INST]"
        with caplog.at_level("WARNING"):
            assert style._messages_to_prompt(messages) == expected
            assert "MistralPromptStyle: First message is from assistant, skipping." in caplog.text

    def test_multiple_assistant_messages_in_a_row(self, style: MistralPromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, "User message"),
            _message(MessageRole.ASSISTANT, "Assistant first response."),
            _message(MessageRole.ASSISTANT, "Assistant second response (after no user message)."),
        ]
        # current_instruction_parts will be empty when processing the second assistant message.
        # The logic is:
        # if current_instruction_parts: prompt += bos_token + "[INST] " + "\n".join(current_instruction_parts) + " [/INST]"
        # prompt += " " + content + "</s>"
        # So, it will correctly append the second assistant message without a new [INST]
        expected = ("<s>[INST] User message [/INST] Assistant first response.</s>"
                    " Assistant second response (after no user message).</s>")
        assert style._messages_to_prompt(messages) == expected

    def test_system_user_assistant_alternating(self, style: MistralPromptStyle) -> None:
        messages = [
            _message(MessageRole.SYSTEM, "System setup."),
            _message(MessageRole.USER, "User query 1."),
            _message(MessageRole.ASSISTANT, "Assistant answer 1."),
            _message(MessageRole.USER, "User query 2."), # System messages are part of INST with user
            _message(MessageRole.ASSISTANT, "Assistant answer 2."),
        ]
        expected = ("<s>[INST] System setup.\nUser query 1. [/INST] Assistant answer 1.</s>"
                    "[INST] User query 2. [/INST] Assistant answer 2.</s>")
        assert style._messages_to_prompt(messages) == expected

    def test_empty_content_messages(self, style: MistralPromptStyle, caplog) -> None:
        messages = [
            _message(MessageRole.USER, "Hello"),
            _message(MessageRole.USER, None), # Skipped by `if not content and message.role != MessageRole.ASSISTANT:`
            _message(MessageRole.USER, ""),   # Skipped by the same logic
            _message(MessageRole.ASSISTANT, "Hi"),
            _message(MessageRole.ASSISTANT, ""), # Empty assistant message, kept
            _message(MessageRole.ASSISTANT, None),# Empty assistant message, kept (content becomes "")
        ]
        # The refactored code skips empty non-assistant messages.
        # Empty assistant messages (content="" or content=None) are kept.
        expected = ("<s>[INST] Hello [/INST] Hi</s>"
                    " </s>" # From assistant with content=""
                    " </s>") # From assistant with content=None (becomes "")

        with caplog.at_level("DEBUG"): # The skipping messages are logged at DEBUG level
             actual = style._messages_to_prompt(messages)
        assert actual == expected
        # Check that specific debug messages for skipping are present
        assert "Skipping empty non-assistant message." in caplog.text # For the None and "" user messages


class TestChatMLPromptStyle:
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    @pytest.fixture
    def style(self) -> ChatMLPromptStyle:
        return ChatMLPromptStyle()

    def test_empty_messages(self, style: ChatMLPromptStyle) -> None:
        messages = []
        # Expected: just the final assistant prompt
        expected = f"{self.IM_START}assistant\n"
        assert style._messages_to_prompt(messages) == expected

    def test_simple_user_assistant_chat(self, style: ChatMLPromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, "Hello"),
            _message(MessageRole.ASSISTANT, "Hi there!"),
        ]
        expected = (
            f"{self.IM_START}user\nHello{self.IM_END}\n"
            f"{self.IM_START}assistant\nHi there!{self.IM_END}\n"
            f"{self.IM_START}assistant\n"
        )
        assert style._messages_to_prompt(messages) == expected

    def test_with_system_message(self, style: ChatMLPromptStyle) -> None:
        messages = [
            _message(MessageRole.SYSTEM, "You are ChatML bot."),
            _message(MessageRole.USER, "Ping"),
            _message(MessageRole.ASSISTANT, "Pong"),
        ]
        expected = (
            f"{self.IM_START}system\nYou are ChatML bot.{self.IM_END}\n"
            f"{self.IM_START}user\nPing{self.IM_END}\n"
            f"{self.IM_START}assistant\nPong{self.IM_END}\n"
            f"{self.IM_START}assistant\n"
        )
        assert style._messages_to_prompt(messages) == expected

    def test_completion_to_prompt(self, style: ChatMLPromptStyle) -> None:
        completion = "Test user input"
        # This will call _messages_to_prompt with [ChatMessage(role=USER, content="Test user input")]
        expected = (
            f"{self.IM_START}user\nTest user input{self.IM_END}\n"
            f"{self.IM_START}assistant\n"
        )
        assert style._completion_to_prompt(completion) == expected

    def test_multiple_turns(self, style: ChatMLPromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, "First user message."),
            _message(MessageRole.ASSISTANT, "First assistant response."),
            _message(MessageRole.USER, "Second user message."),
            _message(MessageRole.ASSISTANT, "Second assistant response."),
        ]
        expected = (
            f"{self.IM_START}user\nFirst user message.{self.IM_END}\n"
            f"{self.IM_START}assistant\nFirst assistant response.{self.IM_END}\n"
            f"{self.IM_START}user\nSecond user message.{self.IM_END}\n"
            f"{self.IM_START}assistant\nSecond assistant response.{self.IM_END}\n"
            f"{self.IM_START}assistant\n"
        )
        assert style._messages_to_prompt(messages) == expected

    def test_message_with_empty_content(self, style: ChatMLPromptStyle) -> None:
        # ChatML typically includes messages even with empty content.
        messages = [
            _message(MessageRole.USER, "Hello"),
            _message(MessageRole.ASSISTANT, ""), # Empty string content
            _message(MessageRole.USER, "Follow up")
        ]
        expected = (
            f"{self.IM_START}user\nHello{self.IM_END}\n"
            f"{self.IM_START}assistant\n{self.IM_END}\n" # Empty content for assistant
            f"{self.IM_START}user\nFollow up{self.IM_END}\n"
            f"{self.IM_START}assistant\n"
        )
        assert style._messages_to_prompt(messages) == expected

    def test_message_with_none_content(self, style: ChatMLPromptStyle) -> None:
        # ChatML typically includes messages even with empty content (None becomes empty string).
        messages = [
            _message(MessageRole.USER, "Hello"),
            _message(MessageRole.ASSISTANT, None), # None content
            _message(MessageRole.USER, "Follow up")
        ]
        expected = (
            f"{self.IM_START}user\nHello{self.IM_END}\n"
            f"{self.IM_START}assistant\n{self.IM_END}\n" # Empty content for assistant
            f"{self.IM_START}user\nFollow up{self.IM_END}\n"
            f"{self.IM_START}assistant\n"
        )
        assert style._messages_to_prompt(messages) == expected

    def test_correct_token_usage_and_newlines(self, style: ChatMLPromptStyle) -> None:
        # Validates: <|im_start|>role\ncontent<|im_end|>\n ... <|im_start|>assistant\n
        messages = [_message(MessageRole.USER, "Test")]
        expected = (
            f"{self.IM_START}user\nTest{self.IM_END}\n"
            f"{self.IM_START}assistant\n"
        )
        actual = style._messages_to_prompt(messages)
        assert actual == expected
        assert actual.count(self.IM_START) == 2
        assert actual.count(self.IM_END) == 1
        assert actual.endswith(f"{self.IM_START}assistant\n")
        # Check newlines: after role, after content (before im_end), after im_end
        # <|im_start|>user\nTest<|im_end|>\n<|im_start|>assistant\n
        # Role is followed by \n. Content is on its own line implicitly. im_end is followed by \n.
        # The structure f"{IM_START}{role}\n{content}{IM_END}\n" ensures this.
        user_part = f"{self.IM_START}user\nTest{self.IM_END}\n"
        assert user_part in actual

        messages_with_system = [
            _message(MessageRole.SYSTEM, "Sys"),
            _message(MessageRole.USER, "Usr")
        ]
        expected_sys_usr = (
            f"{self.IM_START}system\nSys{self.IM_END}\n"
            f"{self.IM_START}user\nUsr{self.IM_END}\n"
            f"{self.IM_START}assistant\n"
        )
        actual_sys_usr = style._messages_to_prompt(messages_with_system)
        assert actual_sys_usr == expected_sys_usr
        assert actual_sys_usr.count(self.IM_START) == 3
        assert actual_sys_usr.count(self.IM_END) == 2


class TestTagPromptStyle:
    BOS = "<s>"
    EOS = "</s>"

    @pytest.fixture
    def style(self) -> TagPromptStyle:
        return TagPromptStyle()

    def test_empty_messages(self, style: TagPromptStyle) -> None:
        messages = []
        # Expected based on current TagPromptStyle: "<s><|assistant|>: "
        expected = f"{self.BOS}<|assistant|>: "
        assert style._messages_to_prompt(messages) == expected

    def test_simple_user_assistant_chat(self, style: TagPromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, "Hello"),
            _message(MessageRole.ASSISTANT, "Hi there!"),
        ]
        expected = (
            f"{self.BOS}<|user|>: Hello\n"
            f"<|assistant|>: Hi there!{self.EOS}\n"
            f"<|assistant|>: "
        )
        assert style._messages_to_prompt(messages) == expected

    def test_with_system_message(self, style: TagPromptStyle) -> None:
        messages = [
            _message(MessageRole.SYSTEM, "System instructions."),
            _message(MessageRole.USER, "Ping"),
            _message(MessageRole.ASSISTANT, "Pong"),
        ]
        expected = (
            f"{self.BOS}<|system|>: System instructions.\n"
            f"<|user|>: Ping\n"
            f"<|assistant|>: Pong{self.EOS}\n"
            f"<|assistant|>: "
        )
        assert style._messages_to_prompt(messages) == expected

    def test_completion_to_prompt(self, style: TagPromptStyle) -> None:
        completion = "Test user input"
        # Expected: <s><|user|>: Test user input\n<|assistant|>:
        expected = f"{self.BOS}<|user|>: Test user input\n<|assistant|>: "
        assert style._completion_to_prompt(completion) == expected

    def test_bos_eos_placement_multiple_turns(self, style: TagPromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, "User1"),
            _message(MessageRole.ASSISTANT, "Assistant1"),
            _message(MessageRole.USER, "User2"),
            _message(MessageRole.ASSISTANT, "Assistant2"),
        ]
        expected = (
            f"{self.BOS}<|user|>: User1\n"
            f"<|assistant|>: Assistant1{self.EOS}\n"
            f"<|user|>: User2\n"
            f"<|assistant|>: Assistant2{self.EOS}\n"
            f"<|assistant|>: "
        )
        assert style._messages_to_prompt(messages) == expected

    def test_ending_with_user_message(self, style: TagPromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, "User1"),
            _message(MessageRole.ASSISTANT, "Assistant1"),
            _message(MessageRole.USER, "User2 (prompting for response)"),
        ]
        expected = (
            f"{self.BOS}<|user|>: User1\n"
            f"<|assistant|>: Assistant1{self.EOS}\n"
            f"<|user|>: User2 (prompting for response)\n"
            f"<|assistant|>: "
        )
        assert style._messages_to_prompt(messages) == expected

    def test_message_with_empty_content(self, style: TagPromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, ""),
            _message(MessageRole.ASSISTANT, ""), # Empty assistant response
        ]
        # Content is stripped, so empty string remains empty.
        expected = (
            f"{self.BOS}<|user|>: \n"
            f"<|assistant|>: {self.EOS}\n"
            f"<|assistant|>: "
        )
        assert style._messages_to_prompt(messages) == expected

    def test_message_with_none_content(self, style: TagPromptStyle) -> None:
        messages = [
            _message(MessageRole.USER, None), # Becomes empty string
            _message(MessageRole.ASSISTANT, None), # Becomes empty string
        ]
        expected = (
            f"{self.BOS}<|user|>: \n"
            f"<|assistant|>: {self.EOS}\n"
            f"<|assistant|>: "
        )
        assert style._messages_to_prompt(messages) == expected

    def test_only_user_message(self, style: TagPromptStyle) -> None:
        messages = [
             _message(MessageRole.USER, "Just a user message"),
        ]
        expected = (
            f"{self.BOS}<|user|>: Just a user message\n"
            f"<|assistant|>: "
        )
        assert style._messages_to_prompt(messages) == expected

    def test_only_assistant_message(self, style: TagPromptStyle) -> None:
        # This is an unusual case, but the style should handle it.
        messages = [
             _message(MessageRole.ASSISTANT, "Only assistant"),
        ]
        expected = (
            f"{self.BOS}<|assistant|>: Only assistant{self.EOS}\n"
            f"<|assistant|>: " # Still prompts for assistant
        )
        assert style._messages_to_prompt(messages) == expected

    def test_only_system_message(self, style: TagPromptStyle) -> None:
        messages = [
             _message(MessageRole.SYSTEM, "Only system"),
        ]
        expected = (
            f"{self.BOS}<|system|>: Only system\n"
            f"<|assistant|>: "
        )
        assert style._messages_to_prompt(messages) == expected
