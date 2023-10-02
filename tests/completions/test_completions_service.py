from llama_index.llms import ChatMessage

from private_gpt.llm.llm_service import LLMService
from tests.fixtures.mock_injector import MockInjector


async def test_completions_service_produces_a_stream(injector: MockInjector) -> None:
    service = injector.get(LLMService)
    stream = await service.stream_complete("test", model_name="mock")
    text = "".join([message.delta or "" async for message in stream])
    assert text == "test"


async def test_completions_service_chat_produces_a_stream(
    injector: MockInjector,
) -> None:
    service = injector.get(LLMService)
    stream = await service.stream_chat([ChatMessage(content="test")], model_name="mock")
    response = "".join([response.delta or "" async for response in stream])
    assert response == "user: test\nassistant: "
