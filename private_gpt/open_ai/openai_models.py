import dataclasses
import json
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

from llama_index.llms import ChatResponse, CompletionResponse


@dataclass(kw_only=True)
class OpenAIDelta:
    """A piece of completion that needs to be concatenated to get the full message."""

    content: str | None


@dataclass(kw_only=True)
class OpenAIMessage:
    """Inference result, with the source of the message.

    Role could be the assistant or system
    (providing a default response, not AI generated).
    """

    role: str
    content: str | None


@dataclass(kw_only=True)
class OpenAIChoice:
    """Response from AI.

    Either the delta or the message will be present, but never both.
    """

    finish_reason: str | None = None
    delta: OpenAIDelta | None = None
    message: OpenAIMessage | None = None
    index: int = 0


@dataclass(kw_only=True)
class OpenAICompletion:
    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIChoice]

    @classmethod
    def from_text(
        cls, text: str | None, finish_reason: str | None = None
    ) -> "OpenAICompletion":
        return OpenAICompletion(
            id=str(uuid.uuid4()),
            object="completion",
            created=int(time.time()),
            model="private-gpt",
            choices=[
                OpenAIChoice(
                    message=OpenAIMessage(role="assistant", content=text),
                    finish_reason=finish_reason,
                )
            ],
        )

    @classmethod
    def json_from_delta(
        cls, *, text: str | None, finish_reason: str | None = None
    ) -> str:
        chunk = OpenAICompletion(
            id=str(uuid.uuid4()),
            object="completion.chunk",
            created=int(time.time()),
            model="private-gpt",
            choices=[
                OpenAIChoice(
                    delta=OpenAIDelta(content=text),
                    finish_reason=finish_reason,
                )
            ],
        )

        return json.dumps(dataclasses.asdict(chunk))


def to_openai_response(response: str | ChatResponse) -> OpenAICompletion:
    if isinstance(response, ChatResponse):
        return OpenAICompletion.from_text(response.delta, finish_reason="stop")
    else:
        return OpenAICompletion.from_text(response, finish_reason="stop")


def to_openai_sse_stream(
    response_generator: Iterator[str | CompletionResponse | ChatResponse],
) -> Iterator[str]:
    for response in response_generator:
        if isinstance(response, CompletionResponse | ChatResponse):
            yield f"data: {OpenAICompletion.json_from_delta(text=response.delta)}\n\n"
        else:
            yield f"data: {OpenAICompletion.json_from_delta(text=response)}\n\n"
    yield f"data: {OpenAICompletion.json_from_delta(text=None, finish_reason='stop')}\n\n"
    yield "data: [DONE]\n\n"
