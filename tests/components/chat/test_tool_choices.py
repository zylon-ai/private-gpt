from llama_index.core.base.llms.types import ChatMessage, MessageRole, TextBlock

from private_gpt.components.chat.processors.chat_history.tools.tool_choices import (
    _add_suffix_to_last_user_message,
)


class TestAddSuffixToLastUserMessage:
    """Tests for _add_suffix_to_last_user_message suffix logic."""

    def test_suffix_with_question_mark(self):
        """Text ending with ? should not get duplicate punctuation."""
        msg = ChatMessage(
            role=MessageRole.USER,
            blocks=[
                TextBlock(text="What are the major steps of ground-water modeling?")
            ],
        )
        result = _add_suffix_to_last_user_message(
            [msg], "Always use one of the available tools to answer your question."
        )
        expected = "What are the major steps of ground-water modeling? Always use one of the available tools to answer your question."
        assert result[0].blocks[0].text == expected

    def test_suffix_with_period(self):
        """Text ending with . should not get duplicate dots."""
        msg = ChatMessage(
            role=MessageRole.USER,
            blocks=[
                TextBlock(text="What are the major steps of ground-water modeling.")
            ],
        )
        result = _add_suffix_to_last_user_message(
            [msg], "Always use one of the available tools to answer your question."
        )
        expected = "What are the major steps of ground-water modeling. Always use one of the available tools to answer your question."
        assert result[0].blocks[0].text == expected

    def test_suffix_with_exclamation(self):
        """Text ending with ! should not get duplicate punctuation."""
        msg = ChatMessage(
            role=MessageRole.USER,
            blocks=[TextBlock(text="Hello world!")],
        )
        result = _add_suffix_to_last_user_message([msg], "Use a tool.")
        expected = "Hello world! Use a tool."
        assert result[0].blocks[0].text == expected

    def test_suffix_without_punctuation(self):
        """Text with no ending punctuation should get a period added."""
        msg = ChatMessage(
            role=MessageRole.USER,
            blocks=[TextBlock(text="Tell me about ground-water modeling")],
        )
        result = _add_suffix_to_last_user_message(
            [msg], "Always use one of the available tools to answer your question."
        )
        expected = "Tell me about ground-water modeling. Always use one of the available tools to answer your question."
        assert result[0].blocks[0].text == expected

    def test_suffix_with_trailing_newline(self):
        """Trailing newlines should be stripped before adding suffix."""
        msg = ChatMessage(
            role=MessageRole.USER,
            blocks=[TextBlock(text="What are the major steps?\n")],
        )
        result = _add_suffix_to_last_user_message(
            [msg], "Always use one of the available tools to answer your question."
        )
        expected = "What are the major steps? Always use one of the available tools to answer your question."
        assert result[0].blocks[0].text == expected

    def test_suffix_with_trailing_spaces(self):
        """Trailing spaces should be stripped before adding suffix."""
        msg = ChatMessage(
            role=MessageRole.USER,
            blocks=[TextBlock(text="Hello world   ")],
        )
        result = _add_suffix_to_last_user_message([msg], "Use a tool.")
        expected = "Hello world. Use a tool."
        assert result[0].blocks[0].text == expected

    def test_suffix_already_present(self):
        """If the suffix is already at the end, the message should not be modified."""
        msg = ChatMessage(
            role=MessageRole.USER,
            blocks=[
                TextBlock(
                    text="Hello. Always use one of the available tools to answer your question."
                )
            ],
        )
        result = _add_suffix_to_last_user_message(
            [msg], "Always use one of the available tools to answer your question."
        )
        assert (
            result[0].blocks[0].text
            == "Hello. Always use one of the available tools to answer your question."
        )

    def test_last_message_is_not_user(self):
        """If the last message is not from the user, no suffix should be added."""
        msg = ChatMessage(role=MessageRole.USER, blocks=[TextBlock(text="User query")])
        assistant_msg = ChatMessage(
            role=MessageRole.ASSISTANT, blocks=[TextBlock(text="Assistant response")]
        )
        result = _add_suffix_to_last_user_message([msg, assistant_msg], "Use a tool.")
        assert result[-1].blocks[0].text == "Assistant response"
        assert result[0].blocks[0].text == "User query"
