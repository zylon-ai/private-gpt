import itertools
import json
from typing import Any

import gradio as gr  # type: ignore
from fastapi import FastAPI
from llama_index.llms import ChatMessage, MessageRole

from private_gpt.di import root_injector
from private_gpt.ingest.ingest_service import IngestService
from private_gpt.llm.llm_service import LLMService
from private_gpt.query.query_service import QueryService
from private_gpt.retrieval.retrieval_service import RetrievalService
from private_gpt.settings import settings

completion_service = root_injector.get(LLMService)
query_service = root_injector.get(QueryService)
ingest_service = root_injector.get(IngestService)
retrieval_service = root_injector.get(RetrievalService)


async def _chat(message: str, history: list[list[str]], mode: str, *_: Any) -> Any:
    stream_response = False
    match mode:
        case "Query Documents":
            stream_response = True
            response = await query_service.stream_chat(message)

        case "LLM Chat":
            history_messages: list[ChatMessage] = list(
                itertools.chain(
                    *[
                        [
                            ChatMessage(content=interaction[0], role=MessageRole.USER),
                            ChatMessage(
                                content=interaction[1], role=MessageRole.ASSISTANT
                            ),
                        ]
                        for interaction in history
                    ]
                )
            )
            # max 20 messages to try to avoid context overflow
            stream_response = True
            response = await completion_service.stream_chat(
                message, history_messages[:20]
            )

        case "Query Chunks":
            stream_response = False
            response = await retrieval_service.retrieve_relevant_nodes(message, 2, 1)

    if stream_response:
        full_response = ""
        async for response_delta in response:
            full_response += response_delta.delta or ""
            yield full_response
    else:
        yield "```" + json.dumps([node.__dict__ for node in response], indent=2)


def _list_ingested_files() -> str:
    files = ingest_service.list()
    return "\n".join(files)


with gr.Blocks() as blocks:
    # Upload button
    file_output = gr.components.Textbox(
        label="Ingested files",
        render=False,
        value=_list_ingested_files,
        every=5,
        lines=5,
        interactive=False,
    )
    upload_button = gr.components.UploadButton(
        "Click to Upload a File", file_count="single", size="sm", render=False
    )
    upload_button.upload(ingest_service.ingest, upload_button)

    # Action dropdown
    dropdown = gr.components.Dropdown(
        choices=["Query Documents", "LLM Chat", "Query Chunks"],
        label="Mode",
        value="Query Documents",
        render=False,
    )

    # Chat
    chat_box = gr.ChatInterface(
        fn=_chat,
        examples=[],
        title="PrivateGPT",
        analytics_enabled=False,
        additional_inputs=[dropdown, upload_button, file_output],
    )


def mount_in_app(app: FastAPI) -> None:
    blocks.queue()
    gr.mount_gradio_app(app, blocks, path=settings.ui.path)
