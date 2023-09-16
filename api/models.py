import dataclasses
import json
import time
import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass(kw_only=True)
class OpenAIDelta:
    content: str


@dataclass(kw_only=True)
class OpenAIChunkChoice:
    delta: Optional[OpenAIDelta]
    finish_reason: str | None
    index: int = 0


@dataclass(kw_only=True)
class OpenAIChunk:
    id: str
    object: str
    created: int
    model: str
    choices: list[OpenAIChunkChoice]

    @classmethod
    def simple_json_chunk(cls, *, text: str | None, finish_reason: str | None = None) -> str:
        chunk = OpenAIChunk(id=str(uuid.uuid4()),
                            object="chat.completion.chunk",
                            created=int(time.time()),
                            model="llama-2",
                            choices=[
                                OpenAIChunkChoice(
                                    delta=OpenAIDelta(content=text),
                                    finish_reason=finish_reason)
                            ])

        return json.dumps(dataclasses.asdict(chunk))
