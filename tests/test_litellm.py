"""Real integration tests for the LiteLLM provider in private-gpt.

Requires ANTHROPIC_FOUNDRY_API_KEY set in the environment.
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

FOUNDRY_KEY = os.environ.get("ANTHROPIC_FOUNDRY_API_KEY", "")
FOUNDRY_BASE = "https://amanrai-test-resource.services.ai.azure.com/anthropic"

requires_key = unittest.skipUnless(FOUNDRY_KEY, "ANTHROPIC_FOUNDRY_API_KEY not set")


def _setup_env():
    os.environ["ANTHROPIC_API_KEY"] = FOUNDRY_KEY
    os.environ["ANTHROPIC_API_BASE"] = FOUNDRY_BASE


@requires_key
class TestLiteLLMComplete(unittest.TestCase):

    def setUp(self):
        _setup_env()
        from private_gpt.components.llm.custom.litellm import LiteLLMCustomLLM
        self.llm = LiteLLMCustomLLM(model="anthropic/claude-sonnet-4-6")

    def test_complete_basic(self):
        resp = self.llm.complete("What is 2+2? Reply with just the number.")
        self.assertIn("4", resp.text)
        self.assertIsNotNone(resp.raw)

    def test_complete_unicode(self):
        resp = self.llm.complete("用中文回答：天空是什么颜色？一个词。")
        self.assertIsNotNone(resp.text)
        self.assertTrue(len(resp.text) > 0)

    def test_complete_long_prompt(self):
        resp = self.llm.complete("Tell me about AI. " * 50 + " One sentence summary.")
        self.assertIsNotNone(resp.text)

    def test_stream_complete(self):
        full = ""
        for chunk in self.llm.stream_complete("Say OK and nothing else."):
            self.assertIsNotNone(chunk.delta)
            full += chunk.delta
        self.assertTrue(len(full) > 0)


@requires_key
class TestLiteLLMChat(unittest.TestCase):

    def setUp(self):
        _setup_env()
        from private_gpt.components.llm.custom.litellm import LiteLLMCustomLLM
        self.llm = LiteLLMCustomLLM(model="anthropic/claude-sonnet-4-6")

    def test_chat_basic(self):
        from llama_index.core.llms import ChatMessage, MessageRole
        messages = [
            ChatMessage(role=MessageRole.USER, content="What is 2+2? Reply with just the number."),
        ]
        resp = self.llm.chat(messages)
        self.assertIn("4", resp.message.content)

    def test_chat_with_system(self):
        from llama_index.core.llms import ChatMessage, MessageRole
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="Always reply in exactly one word."),
            ChatMessage(role=MessageRole.USER, content="What color is the sky?"),
        ]
        resp = self.llm.chat(messages)
        self.assertIsNotNone(resp.message.content)

    def test_chat_multi_turn(self):
        from llama_index.core.llms import ChatMessage, MessageRole
        messages = [
            ChatMessage(role=MessageRole.USER, content="My name is Alice."),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hello Alice!"),
            ChatMessage(role=MessageRole.USER, content="What is my name? Reply with just the name."),
        ]
        resp = self.llm.chat(messages)
        self.assertIn("Alice", resp.message.content)

    def test_stream_chat(self):
        from llama_index.core.llms import ChatMessage, MessageRole
        messages = [
            ChatMessage(role=MessageRole.USER, content="Say OK."),
        ]
        full = ""
        for chunk in self.llm.stream_chat(messages):
            if chunk.delta:
                full += chunk.delta
        self.assertTrue(len(full) > 0)


@requires_key
class TestLiteLLMParameterForwarding(unittest.TestCase):

    def setUp(self):
        _setup_env()

    def test_max_tokens_respected(self):
        from private_gpt.components.llm.custom.litellm import LiteLLMCustomLLM
        llm = LiteLLMCustomLLM(model="anthropic/claude-sonnet-4-6", max_new_tokens=5)
        resp = llm.complete("Write a very long essay about the history of computing.")
        self.assertTrue(len(resp.text) < 200)

    def test_temperature_forwarded(self):
        from private_gpt.components.llm.custom.litellm import LiteLLMCustomLLM
        llm = LiteLLMCustomLLM(model="anthropic/claude-sonnet-4-6", temperature=0.0)
        resp = llm.complete("What is 2+2? Reply with just the number.")
        self.assertIn("4", resp.text)

    def test_custom_context_window(self):
        from private_gpt.components.llm.custom.litellm import LiteLLMCustomLLM
        llm = LiteLLMCustomLLM(model="anthropic/claude-sonnet-4-6", context_window=8192)
        self.assertEqual(llm.metadata.context_window, 8192)

    def test_custom_timeout(self):
        from private_gpt.components.llm.custom.litellm import LiteLLMCustomLLM
        llm = LiteLLMCustomLLM(model="anthropic/claude-sonnet-4-6", request_timeout=5.0)
        resp = llm.complete("Say OK.")
        self.assertIsNotNone(resp.text)


@requires_key
class TestLiteLLMEdgeCases(unittest.TestCase):

    def setUp(self):
        _setup_env()

    def test_nonexistent_model(self):
        from private_gpt.components.llm.custom.litellm import LiteLLMCustomLLM
        llm = LiteLLMCustomLLM(model="anthropic/nonexistent-model-xyz")
        with self.assertRaises(Exception):
            llm.complete("test")

    def test_metadata(self):
        from private_gpt.components.llm.custom.litellm import LiteLLMCustomLLM
        llm = LiteLLMCustomLLM(model="anthropic/claude-sonnet-4-6")
        meta = llm.metadata
        self.assertEqual(meta.model_name, "anthropic/claude-sonnet-4-6")
        self.assertGreater(meta.context_window, 0)
        self.assertGreater(meta.num_output, 0)

    def test_empty_prompt_raises(self):
        from private_gpt.components.llm.custom.litellm import LiteLLMCustomLLM
        llm = LiteLLMCustomLLM(model="anthropic/claude-sonnet-4-6")
        with self.assertRaises(Exception):
            llm.complete("")

    def test_raw_response_has_expected_keys(self):
        from private_gpt.components.llm.custom.litellm import LiteLLMCustomLLM
        llm = LiteLLMCustomLLM(model="anthropic/claude-sonnet-4-6")
        resp = llm.complete("Say OK.")
        self.assertIn("choices", resp.raw)
        self.assertIn("model", resp.raw)

    def test_stream_complete_accumulates_text(self):
        from private_gpt.components.llm.custom.litellm import LiteLLMCustomLLM
        llm = LiteLLMCustomLLM(model="anthropic/claude-sonnet-4-6")
        chunks = list(llm.stream_complete("Count from 1 to 3."))
        if len(chunks) > 1:
            self.assertTrue(len(chunks[-1].text) >= len(chunks[0].text))

    def test_chat_role_conversion(self):
        from private_gpt.components.llm.custom.litellm import LiteLLMCustomLLM
        from llama_index.core.llms import ChatMessage, MessageRole
        llm = LiteLLMCustomLLM(model="anthropic/claude-sonnet-4-6")
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are a math tutor."),
            ChatMessage(role=MessageRole.USER, content="What is 3*3?"),
            ChatMessage(role=MessageRole.ASSISTANT, content="9"),
            ChatMessage(role=MessageRole.USER, content="And 4*4?"),
        ]
        resp = llm.chat(messages)
        self.assertIn("16", resp.message.content)


if __name__ == "__main__":
    unittest.main()
