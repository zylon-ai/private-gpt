import dataclasses
import json
import time
import uuid
from dataclasses import dataclass
from typing import Optional


# {
#     "id": "chatcmpl-123",
#     "object": "chat.completion",
#     "created": 1677652288,
#     "model": "gpt-3.5-turbo-0613",
#     "choices": [{
#         "index": 0,
#         "message": {
#             "role": "assistant",
#             "content": "\n\nHello there, how may I assist you today?",
#         },
#         "finish_reason": "stop"
#     }],
#     "usage": {
#         "prompt_tokens": 9,
#         "completion_tokens": 12,
#         "total_tokens": 21
#     }
# }

# {
#     "id": "chatcmpl-123",
#     "object": "chat.completion.chunk",
#     "created": 1677652288,
#     "model": "gpt-3.5-turbo",
#     "choices": [{
#         "index": 0,
#         "delta": {
#             "content": "Hello",
#         },
#         "finish_reason": "stop"
#     }]
# }


@dataclass(kw_only=True)
class OpenAIDelta:
    """
    A piece of completion that needs to be concatenated to get the full message.
    """
    content: str


@dataclass(kw_only=True)
class OpenAIMessage:
    """
    Inference result, with the source of the message.
    Role could be the assistant or system (providing a default response, not AI generated)
    """
    role: str
    content: str


@dataclass(kw_only=True)
class OpenAIChoice:
    """
    When AI is prompted, it can provide several options,
    either the delta or the message will be present, but never both.
    """
    finish_reason: str | None = None
    delta: Optional[OpenAIDelta] = None
    message: Optional[OpenAIMessage] = None
    index: int = 0


@dataclass(kw_only=True)
class OpenAICompletion:
    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIChoice]

    @classmethod
    def simple_json_delta(cls, *, text: str | None, finish_reason: str | None = None) -> str:
        chunk = OpenAICompletion(id=str(uuid.uuid4()),
                                 object="chat.completion.chunk",
                                 created=int(time.time()),
                                 model="llama-2",
                                 choices=[
                                     OpenAIChoice(
                                         delta=OpenAIDelta(content=text),
                                         finish_reason=finish_reason)
                                 ])

        return json.dumps(dataclasses.asdict(chunk))
