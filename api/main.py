import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llama_index.llms import LlamaCPP
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from pydantic import BaseModel

from api.models import OpenAICompletion

llms = {}

PORT = int(os.environ.get("PORT") or "8001")


@asynccontextmanager
async def lifespan(app: FastAPI):
    models_folder = Path(os.path.dirname(__file__)).parent.joinpath("models")
    llms["llama"] = LlamaCPP(
        # model_url="https://huggingface.co/TheBloke/Llama-2-7B-chat-GGUF/resolve/main/llama-2-7b-chat.Q4_0.gguf",
        model_path=f"{models_folder.absolute()}/llama-2-7b-chat.Q4_0.gguf",
        temperature=0.1,
        max_new_tokens=256,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 1},
        # transform inputs into Llama2 format
        messages_to_prompt=messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )
    yield


app = FastAPI(lifespan=lifespan)


def run_llm(prompt: str) -> AsyncGenerator:
    llm: LlamaCPP = llms["llama"]
    truncated_prompt = prompt[:llm.context_window]
    response_iter = llm.stream_complete(truncated_prompt)
    for response in response_iter:
        yield f"data: {OpenAICompletion.simple_json_delta(text=response.delta)}\n\n"
    yield f"data: [DONE]\n\n"


@app.get("/health")
def root() -> str:
    return "ok"


class InferenceBody(BaseModel):
    prompt: str


@app.post("/")
async def inference(body: InferenceBody) -> StreamingResponse:
    return StreamingResponse(run_llm(body.prompt), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
