import dataclasses
import json
import time
import uuid
from dataclasses import dataclass


@dataclass(kw_only=True)
class OpenAIDelta:
    """A piece of completion that needs to be concatenated to get the full message."""

    content: str


@dataclass(kw_only=True)
class OpenAIMessage:
    """Inference result, with the source of the message.

    Role could be the assistant or system
    (providing a default response, not AI generated).
    """

    role: str
    content: str


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
    def simple_json_delta(
        cls, *, text: str | None, finish_reason: str | None = None
    ) -> str:
        chunk = OpenAICompletion(
            id=str(uuid.uuid4()),
            object="chat.completion.chunk",
            created=int(time.time()),
            model="llama-2",
            choices=[
                OpenAIChoice(
                    delta=OpenAIDelta(content=text), finish_reason=finish_reason
                )
            ],
        )

        return json.dumps(dataclasses.asdict(chunk))
