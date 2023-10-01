import itertools
from typing import Any

import gradio as gr  # type: ignore
from fastapi import FastAPI
from llama_index.llms import ChatMessage, MessageRole

from private_gpt.completions.completions_service import CompletionsService
from private_gpt.di import root_injector
from private_gpt.settings import settings

completion_service = root_injector.get(CompletionsService)


def _chat(message: str, history: list[list[str]]) -> Any:
    history_messages: list[ChatMessage] = list(
        itertools.chain(
            *[
                [
                    ChatMessage(content=interaction[0], role=MessageRole.USER),
                    ChatMessage(content=interaction[1], role=MessageRole.ASSISTANT),
                ]
                for interaction in history
            ]
        )
    )
    new_message = ChatMessage(content=message, role=MessageRole.USER)
    # max 20 messages to try to avoid context overflow
    messages = [*history_messages[:20], new_message]
    response = completion_service.stream_chat(messages)
    full_response = ""
    for response_delta in response:
        full_response += response_delta.delta or ""
        yield full_response


with gr.Blocks() as blocks:
    chat_box = gr.ChatInterface(
        fn=_chat,
        examples=[],
        title="PrivateGPT",
    )


def mount_in_app(app: FastAPI) -> None:
    blocks.queue()
    gr.mount_gradio_app(app, blocks, path=settings.ui.path)
