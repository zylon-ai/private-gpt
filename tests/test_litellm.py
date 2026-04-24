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


if __name__ == "__main__":
    unittest.main()
