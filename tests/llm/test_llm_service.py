
from private_gpt.llm.llm_service import LLMService
from tests.fixtures.mock_injector import MockInjector


async def test_llm_service_produces_a_stream(injector: MockInjector) -> None:
    service = injector.get(LLMService)
    stream = await service.stream_complete("test")
    text = "".join([message.delta or "" async for message in stream])
    assert text == "test"


async def test_llm_service_chat_produces_a_stream(
    injector: MockInjector,
) -> None:
    service = injector.get(LLMService)
    stream = await service.stream_chat("test")
    response = "".join([response.delta or "" async for response in stream])
    assert response == "user: test\nassistant: "
