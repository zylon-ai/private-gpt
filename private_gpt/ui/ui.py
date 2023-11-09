import itertools
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from llama_index.llms import ChatMessage, ChatResponse, MessageRole

from private_gpt.di import root_injector
from private_gpt.server.chat.chat_service import ChatService
from private_gpt.server.chunks.chunks_service import ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.settings.settings import settings
from private_gpt.ui.images import logo_svg

ingest_service = root_injector.get(IngestService)
chat_service = root_injector.get(ChatService)
chunks_service = root_injector.get(ChunksService)


def _chat(message: str, history: list[list[str]], mode: str, *_: Any) -> Any:
    def yield_deltas(stream: Iterable[ChatResponse | str]) -> Iterable[str]:
        full_response: str = ""
        for delta in stream:
            if isinstance(delta, str):
                full_response += str(delta)
            elif isinstance(delta, ChatResponse):
                full_response += delta.delta or ""
            yield full_response

    def build_history() -> list[ChatMessage]:
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

        # max 20 messages to try to avoid context overflow
        return history_messages[:20]

    new_message = ChatMessage(content=message, role=MessageRole.USER)
    all_messages = [*build_history(), new_message]
    match mode:
        case "Query Docs":
            query_stream = chat_service.stream_chat(
                messages=all_messages,
                use_context=True,
            )
            yield from yield_deltas(query_stream)

        case "LLM Chat":
            llm_stream = chat_service.stream_chat(
                messages=all_messages,
                use_context=False,
            )
            yield from yield_deltas(llm_stream)

        case "Search in Docs":
            response = chunks_service.retrieve_relevant(
                text=message, limit=4, prev_next_chunks=0
            )

            yield "\n\n\n".join(
                f"{index}. **{chunk.document.doc_metadata['file_name'] if chunk.document.doc_metadata else ''} "
                f"(page {chunk.document.doc_metadata['page_label'] if chunk.document.doc_metadata else ''})**\n "
                f"{chunk.text}"
                for index, chunk in enumerate(response, start=1)
            )


def _list_ingested_files() -> list[str]:
    files = set()
    for ingested_document in ingest_service.list_ingested():
        if ingested_document.doc_metadata is not None:
            files.add(
                ingested_document.doc_metadata.get("file_name") or "[FILE NAME MISSING]"
            )
    return list(files)


# Global state
_uploaded_file_list = [[row] for row in _list_ingested_files()]


def _upload_file(file: TextIO) -> list[list[str]]:
    path = Path(file.name)
    ingest_service.ingest(file_name=path.name, file_data=path)
    _uploaded_file_list.append([path.name])
    return _uploaded_file_list


with gr.Blocks(
    theme=gr.themes.Soft(primary_hue=slate),
    css=".logo { "
    "display:flex;"
    "background-color: #C7BAFF;"
    "height: 80px;"
    "border-radius: 8px;"
    "align-content: center;"
    "justify-content: center;"
    "align-items: center;"
    "}"
    ".logo img { height: 25% }",
) as blocks:
    with gr.Row():
        gr.HTML(f"<div class='logo'/><img src={logo_svg} alt=PrivateGPT></div")

    with gr.Row():
        with gr.Column(scale=3, variant="compact"):
            mode = gr.Radio(
                ["Query Docs", "Search in Docs", "LLM Chat"],
                label="Mode",
                value="Query Docs",
            )
            upload_button = gr.components.UploadButton(
                "Upload a File",
                type="file",
                file_count="single",
                size="sm",
            )
            ingested_dataset = gr.List(
                _uploaded_file_list,
                headers=["File name"],
                label="Ingested Files",
                interactive=False,
                render=False,  # Rendered under the button
            )
            upload_button.upload(
                _upload_file, inputs=upload_button, outputs=ingested_dataset
            )
            ingested_dataset.render()
        with gr.Column(scale=7):
            chatbot = gr.ChatInterface(
                _chat,
                chatbot=gr.Chatbot(
                    label=f"LLM: {settings.llm.mode}",
                    show_copy_button=True,
                    render=False,
                    avatar_images=(
                        None,
                        "https://lh3.googleusercontent.com/drive-viewer/AK7aPa"
                        "AicXck0k68nsscyfKrb18o9ak3BSaWM_Qzm338cKoQlw72Bp0UKN84"
                        "IFZjXjZApY01mtnUXDeL4qzwhkALoe_53AhwCg=s2560",
                    ),
                ),
                additional_inputs=[mode, upload_button],
            )


def mount_in_app(app: FastAPI) -> None:
    blocks.queue()
    gr.mount_gradio_app(app, blocks, path=settings.ui.path)


if __name__ == "__main__":
    blocks.queue()
    blocks.launch(debug=False, show_api=False)
