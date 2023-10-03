import openai.openai_response
from fastapi.testclient import TestClient
from openai.api_requestor import parse_stream

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


async def test_llm_endpoint_produces_sse_stream(test_client: TestClient) -> None:
    response = test_client.get("/completions?prompt=test")

    raw_events = response.text.split("\n\n")
    events = [
        item.removeprefix("data: ") for item in raw_events if item.startswith("data: ")
    ]
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert len(events) > 0
    assert events[-1] == "[DONE]"
