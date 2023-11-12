from fastapi.testclient import TestClient

from private_gpt.open_ai.openai_models import OpenAICompletion, OpenAIMessage
from private_gpt.server.chat.chat_router import ChatBody


def test_chat_route_produces_a_stream(test_client: TestClient) -> None:
    body = ChatBody(
        messages=[OpenAIMessage(content="test", role="user")],
        use_context=False,
        stream=True,
    )
    response = test_client.post("/v1/chat/completions", json=body.model_dump())

    raw_events = response.text.split("\n\n")
    events = [
        item.removeprefix("data: ") for item in raw_events if item.startswith("data: ")
    ]
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    assert len(events) > 0
    assert events[-1] == "[DONE]"


def test_chat_route_produces_a_single_value(test_client: TestClient) -> None:
    body = ChatBody(
        messages=[OpenAIMessage(content="test", role="user")],
        use_context=False,
        stream=False,
    )
    response = test_client.post("/v1/chat/completions", json=body.model_dump())

    # No asserts, if it validates it's good
    OpenAICompletion.model_validate(response.json())
    assert response.status_code == 200
