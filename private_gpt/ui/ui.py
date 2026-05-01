"""This file should be imported if and only if you want to run the UI locally."""

import base64
import functools
import logging
import shutil
import subprocess
import time
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from injector import inject, singleton
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from llama_index.core.types import TokenGen
from pydantic import BaseModel

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.paths import local_data_path
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.incremental_ingest_service import (
    IncrementalIngestService,
)
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.recipes.summarize.summarize_service import SummarizeService
from private_gpt.settings.settings import settings
from private_gpt.ui.images import logo_svg

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
# Should be "private_gpt/ui/avatar-bot.ico"
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "avatar-bot.ico"

UI_TAB_TITLE = "My Private GPT"

SOURCES_SEPARATOR = "<hr>Sources: \n"


class Modes(str, Enum):
    RAG_MODE = "RAG"
    SEARCH_MODE = "Search"
    BASIC_CHAT_MODE = "Basic"
    SUMMARIZE_MODE = "Summarize"


MODES: list[Modes] = [
    Modes.RAG_MODE,
    Modes.SEARCH_MODE,
    Modes.BASIC_CHAT_MODE,
    Modes.SUMMARIZE_MODE,
]


class Source(BaseModel):
    file: str
    page: str
    text: str

    class Config:
        frozen = True

    @staticmethod
    def curate_sources(sources: list[Chunk]) -> list["Source"]:
        curated_sources = []

        for chunk in sources:
            doc_metadata = chunk.document.doc_metadata

            file_name = doc_metadata.get("file_name", "-") if doc_metadata else "-"
            page_label = doc_metadata.get("page_label", "-") if doc_metadata else "-"

            source = Source(file=file_name, page=page_label, text=chunk.text)
            curated_sources.append(source)
            curated_sources = list(
                dict.fromkeys(curated_sources).keys()
            )  # Unique sources only

        return curated_sources


@singleton
class PrivateGptUi:
    @inject
    def __init__(
        self,
        ingest_service: IngestService,
        chat_service: ChatService,
        chunks_service: ChunksService,
        summarizeService: SummarizeService,
        incremental_ingest_service: IncrementalIngestService,
    ) -> None:
        self._ingest_service = ingest_service
        self._chat_service = chat_service
        self._chunks_service = chunks_service
        self._summarize_service = summarizeService
        self._incremental_svc = incremental_ingest_service

        # Cache the UI blocks
        self._ui_block = None

        self._selected_filename = None

        # Incremental mode is configured via settings.yaml (incremental.enabled),
        # not toggled at runtime — the IngestService component is a singleton.
        self._incremental_enabled = settings().incremental.enabled or (
            settings().embedding.ingest_mode == "incremental"
        )

        # Initialize system prompt based on default mode
        default_mode_map = {mode.value: mode for mode in Modes}
        self._default_mode = default_mode_map.get(
            settings().ui.default_mode, Modes.RAG_MODE
        )
        self._system_prompt = self._get_default_system_prompt(self._default_mode)

    def _chat(
        self, message: str, history: list[list[str]], mode: Modes, *_: Any
    ) -> Any:
        def yield_deltas(completion_gen: CompletionGen) -> Iterable[str]:
            full_response: str = ""
            stream = completion_gen.response
            for delta in stream:
                if isinstance(delta, str):
                    full_response += str(delta)
                elif isinstance(delta, ChatResponse):
                    full_response += delta.delta or ""
                yield full_response
                time.sleep(0.02)

            if completion_gen.sources:
                full_response += SOURCES_SEPARATOR
                cur_sources = Source.curate_sources(completion_gen.sources)
                sources_text = "\n\n\n"
                used_files = set()
                for index, source in enumerate(cur_sources, start=1):
                    if f"{source.file}-{source.page}" not in used_files:
                        sources_text = (
                            sources_text
                            + f"{index}. {source.file} (page {source.page}) \n\n"
                        )
                        used_files.add(f"{source.file}-{source.page}")
                sources_text += "<hr>\n\n"
                full_response += sources_text
            yield full_response

        def yield_tokens(token_gen: TokenGen) -> Iterable[str]:
            full_response: str = ""
            for token in token_gen:
                full_response += str(token)
                yield full_response

        def build_history() -> list[ChatMessage]:
            history_messages: list[ChatMessage] = []

            for interaction in history:
                history_messages.append(
                    ChatMessage(content=interaction[0], role=MessageRole.USER)
                )
                if len(interaction) > 1 and interaction[1] is not None:
                    history_messages.append(
                        ChatMessage(
                            # Remove from history content the Sources information
                            content=interaction[1].split(SOURCES_SEPARATOR)[0],
                            role=MessageRole.ASSISTANT,
                        )
                    )

            # max 20 messages to try to avoid context overflow
            return history_messages[:20]

        new_message = ChatMessage(content=message, role=MessageRole.USER)
        all_messages = [*build_history(), new_message]
        # If a system prompt is set, add it as a system message
        if self._system_prompt:
            all_messages.insert(
                0,
                ChatMessage(
                    content=self._system_prompt,
                    role=MessageRole.SYSTEM,
                ),
            )
        match mode:
            case Modes.RAG_MODE:
                # Use only the selected file for the query
                context_filter = None
                if self._selected_filename is not None:
                    docs_ids = []
                    for ingested_document in self._ingest_service.list_ingested():
                        if (
                            ingested_document.doc_metadata["file_name"]
                            == self._selected_filename
                        ):
                            docs_ids.append(ingested_document.doc_id)
                    context_filter = ContextFilter(docs_ids=docs_ids)

                query_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=True,
                    context_filter=context_filter,
                )
                yield from yield_deltas(query_stream)
            case Modes.BASIC_CHAT_MODE:
                llm_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=False,
                )
                yield from yield_deltas(llm_stream)

            case Modes.SEARCH_MODE:
                response = self._chunks_service.retrieve_relevant(
                    text=message, limit=4, prev_next_chunks=0
                )

                sources = Source.curate_sources(response)

                yield "\n\n\n".join(
                    f"{index}. **{source.file} "
                    f"(page {source.page})**\n "
                    f"{source.text}"
                    for index, source in enumerate(sources, start=1)
                )
            case Modes.SUMMARIZE_MODE:
                # Summarize the given message, optionally using selected files
                context_filter = None
                if self._selected_filename:
                    docs_ids = []
                    for ingested_document in self._ingest_service.list_ingested():
                        if (
                            ingested_document.doc_metadata["file_name"]
                            == self._selected_filename
                        ):
                            docs_ids.append(ingested_document.doc_id)
                    context_filter = ContextFilter(docs_ids=docs_ids)

                summary_stream = self._summarize_service.stream_summarize(
                    use_context=True,
                    context_filter=context_filter,
                    instructions=message,
                )
                yield from yield_tokens(summary_stream)

    # On initialization and on mode change, this function set the system prompt
    # to the default prompt based on the mode (and user settings).
    @staticmethod
    def _get_default_system_prompt(mode: Modes) -> str:
        p = ""
        match mode:
            # For query chat mode, obtain default system prompt from settings
            case Modes.RAG_MODE:
                p = settings().ui.default_query_system_prompt
            # For chat mode, obtain default system prompt from settings
            case Modes.BASIC_CHAT_MODE:
                p = settings().ui.default_chat_system_prompt
            # For summarization mode, obtain default system prompt from settings
            case Modes.SUMMARIZE_MODE:
                p = settings().ui.default_summarization_system_prompt
            # For any other mode, clear the system prompt
            case _:
                p = ""
        return p

    @staticmethod
    def _get_default_mode_explanation(mode: Modes) -> str:
        match mode:
            case Modes.RAG_MODE:
                return "Get contextualized answers from selected files."
            case Modes.SEARCH_MODE:
                return "Find relevant chunks of text in selected files."
            case Modes.BASIC_CHAT_MODE:
                return "Chat with the LLM using its training data. Files are ignored."
            case Modes.SUMMARIZE_MODE:
                return "Generate a summary of the selected files. Prompt to customize the result."
            case _:
                return ""

    def _set_system_prompt(self, system_prompt_input: str) -> None:
        logger.info(f"Setting system prompt to: {system_prompt_input}")
        self._system_prompt = system_prompt_input

    def _set_explanatation_mode(self, explanation_mode: str) -> None:
        self._explanation_mode = explanation_mode

    def _set_current_mode(self, mode: Modes) -> Any:
        self.mode = mode
        self._set_system_prompt(self._get_default_system_prompt(mode))
        self._set_explanatation_mode(self._get_default_mode_explanation(mode))
        interactive = self._system_prompt is not None
        return [
            gr.update(placeholder=self._system_prompt, interactive=interactive),
            gr.update(value=self._explanation_mode),
        ]

    def _list_ingested_files(self) -> list[list[str]]:
        files = set()
        for ingested_document in self._ingest_service.list_ingested():
            if ingested_document.doc_metadata is None:
                # Skipping documents without metadata
                continue
            file_name = ingested_document.doc_metadata.get(
                "file_name", "[FILE NAME MISSING]"
            )
            files.add(file_name)
        return [[row] for row in files]

    def _upload_file(self, files: list[str]) -> list[list[str]]:
        logger.debug("Loading count=%s files", len(files))
        paths = [Path(file) for file in files]

        if not self._incremental_enabled:
            # Non-incremental: remove all existing Documents with name
            # identical to a new file upload, then re-ingest from scratch.
            file_names = [path.name for path in paths]
            doc_ids_to_delete = []
            for ingested_document in self._ingest_service.list_ingested():
                if (
                    ingested_document.doc_metadata
                    and ingested_document.doc_metadata["file_name"] in file_names
                ):
                    doc_ids_to_delete.append(ingested_document.doc_id)
            if doc_ids_to_delete:
                logger.info(
                    "Uploading file(s) which were already ingested: "
                    "%s document(s) will be replaced.",
                    len(doc_ids_to_delete),
                )
                for doc_id in doc_ids_to_delete:
                    self._ingest_service.delete(doc_id)
            self._ingest_service.bulk_ingest([(str(path.name), path) for path in paths])
            return self._list_ingested_files()

        # Incremental mode: always persist to a stable location AND register
        # with the watcher. Browser uploads only expose Gradio tmp paths
        # (cleaned up after the request), so we need a persistent copy for
        # the watcher to monitor. Auto-start the watcher if it isn't running —
        # the user shouldn't have to touch the watcher panel manually.
        if not self._incremental_svc.is_watcher_running:
            self._incremental_svc.start_watching_background()

        uploads_dir = local_data_path / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        for tmp_path in paths:
            persistent = uploads_dir / tmp_path.name
            # Pre-arm debounce BEFORE writing so the filesystem event from
            # shutil.copy2 is suppressed for already-watched files.
            self._incremental_svc.touch_debounce(persistent)
            shutil.copy2(tmp_path, persistent)
            logger.info("Saved upload to %s and registered for watching", persistent)
            self._incremental_svc.ingest_file_from_path(persistent)
        return self._list_ingested_files()

    def _delete_all_files(self) -> Any:
        ingested_files = self._ingest_service.list_ingested()
        logger.debug("Deleting count=%s files", len(ingested_files))
        for ingested_document in ingested_files:
            self._ingest_service.delete(ingested_document.doc_id)
        return [
            gr.List(self._list_ingested_files()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _delete_selected_file(self) -> Any:
        logger.debug("Deleting selected %s", self._selected_filename)
        # Note: keep looping for pdf's (each page became a Document)
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata["file_name"]
                == self._selected_filename
            ):
                self._ingest_service.delete(ingested_document.doc_id)
        return [
            gr.List(self._list_ingested_files()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _deselect_selected_file(self) -> Any:
        self._selected_filename = None
        return [
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _selected_a_file(self, select_data: gr.SelectData) -> Any:
        self._selected_filename = select_data.value
        return [
            gr.components.Button(interactive=True),
            gr.components.Button(interactive=True),
            gr.components.Textbox(self._selected_filename),
        ]

    # ── Watcher control helpers ────────────────────────────────────────

    def _watcher_status_text(self) -> str:
        """One-line markdown status for the watcher panel."""
        paths = self._incremental_svc.watched_file_paths
        n = len(paths)
        if self._incremental_svc.is_watcher_running:
            return f"**Status:** Running — {n} file(s) registered"
        return f"**Status:** Stopped — {n} file(s) registered (will resume on Start)"

    def _watched_files_text(self) -> str:
        """Newline-separated list of watched file paths for the textbox."""
        paths = self._incremental_svc.watched_file_paths
        return "\n".join(str(p) for p in paths) if paths else "(none)"

    def _stop_watcher(self) -> tuple[str, str, str, str]:
        """Stop the observer thread."""
        if self._incremental_svc.is_watcher_running:
            self._incremental_svc.stop_watching()
        return (
            self._watcher_status_text(),
            gr.update(interactive=True),  # start button
            gr.update(interactive=False),  # stop button
            self._watched_files_text(),
        )

    def _start_watcher(self) -> tuple[str, str, str, str]:
        """Restart the observer thread and re-watch all registered files."""
        if not self._incremental_svc.is_watcher_running:
            self._incremental_svc.start_watching_background()
        return (
            self._watcher_status_text(),
            gr.update(interactive=False),  # start button
            gr.update(interactive=True),  # stop button
            self._watched_files_text(),
        )

    def _unwatch_all(self) -> tuple[str, str, str, str]:
        """Unregister all watched files without stopping the observer."""
        self._incremental_svc.unwatch_all_files()
        return (
            self._watcher_status_text(),
            gr.update(interactive=not self._incremental_svc.is_watcher_running),
            gr.update(interactive=self._incremental_svc.is_watcher_running),
            self._watched_files_text(),
        )

    def _watch_path(self, path_str: str) -> tuple[Any, str, str, str]:
        """Register an absolute filesystem path with the watcher.

        The file is ingested incrementally (full ingest on first sight,
        chunk-level diff on subsequent saves) and the original location
        is watched — editing the file in place triggers an auto-update.
        """
        if not path_str or not path_str.strip():
            message = "Enter an absolute path to a file."
        else:
            path = Path(path_str.strip()).expanduser()
            try:
                if not path.exists():
                    message = f"File not found: {path}"
                elif not path.is_file():
                    message = f"Not a regular file: {path}"
                else:
                    if not self._incremental_svc.is_watcher_running:
                        self._incremental_svc.start_watching_background()
                    stats = self._incremental_svc.ingest_file_from_path(path)
                    message = (
                        f"Watching {path}\n"
                        f"Chunks: {stats.total_chunks_new} "
                        f"(modified={stats.chunks_modified}, "
                        f"added={stats.chunks_added}, "
                        f"unchanged={stats.chunks_unchanged})"
                    )
            except Exception as exc:
                message = f"Error: {exc}"
        return (
            gr.List(self._list_ingested_files()),
            self._watcher_status_text(),
            self._watched_files_text(),
            message,
        )

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
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
            ".logo img { height: 25% }"
            ".contain { display: flex !important; flex-direction: column !important; }"
            "#component-0, #component-3, #component-10, #component-8  { height: 100% !important; }"
            "#chatbot { flex-grow: 1 !important; overflow: auto !important;}"
            "#col { height: calc(100vh - 112px - 16px) !important; }"
            "hr { margin-top: 1em; margin-bottom: 1em; border: 0; border-top: 1px solid #FFF; }"
            ".avatar-image { background-color: antiquewhite; border-radius: 2px; }"
            ".footer { text-align: center; margin-top: 20px; font-size: 14px; display: flex; align-items: center; justify-content: center; }"
            ".footer-zylon-link { display:flex; margin-left: 5px; text-decoration: auto; color: var(--body-text-color); }"
            ".footer-zylon-link:hover { color: #C7BAFF; }"
            ".footer-zylon-ico { height: 20px; margin-left: 5px; background-color: antiquewhite; border-radius: 2px; }",
        ) as blocks:
            with gr.Row():
                gr.HTML(f"<div class='logo'/><img src={logo_svg} alt=PrivateGPT></div")

            # ── Helper: resolve model label ────────────────────────
            def get_model_label() -> str | None:
                config_settings = settings()
                if config_settings is None:
                    raise ValueError("Settings are not configured.")
                llm_mode = config_settings.llm.mode
                model_mapping = {
                    "llamacpp": config_settings.llamacpp.llm_hf_model_file,
                    "openai": config_settings.openai.model,
                    "openailike": config_settings.openai.model,
                    "azopenai": config_settings.azopenai.llm_model,
                    "sagemaker": config_settings.sagemaker.llm_endpoint_name,
                    "mock": llm_mode,
                    "ollama": config_settings.ollama.llm_model,
                    "gemini": config_settings.gemini.model,
                }
                if llm_mode not in model_mapping:
                    print(f"Invalid 'llm mode': {llm_mode}")
                    return None
                return model_mapping[llm_mode]

            model_label = get_model_label()
            label_text = (
                f"LLM: {settings().llm.mode} | Model: {model_label}"
                if model_label is not None
                else f"LLM: {settings().llm.mode}"
            )

            # ── Tabs ────────────────────────────────────────────────
            with gr.Tabs():
                # ═══════════════ Tab 1: Chat & Ingest ═══════════════
                with gr.Tab("Chat & Ingest"):  # noqa: SIM117 - Gradio layout container
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=3):
                            default_mode = self._default_mode
                            mode = gr.Radio(
                                [m.value for m in MODES],
                                label="Mode",
                                value=default_mode,
                            )
                            explanation_mode = gr.Textbox(
                                placeholder=self._get_default_mode_explanation(
                                    default_mode
                                ),
                                show_label=False,
                                max_lines=3,
                                interactive=False,
                            )
                            # Defined early (render=False) so the upload
                            # event can reference them before they appear
                            # in the layout below.
                            _watcher_status_cmp = gr.Markdown(
                                value=self._watcher_status_text(),
                                render=False,
                            )
                            _watcher_files_cmp = gr.Textbox(
                                label="Watched files (updated automatically on upload)",
                                value=self._watched_files_text(),
                                lines=3,
                                interactive=False,
                                render=False,
                            )

                            upload_button = gr.components.UploadButton(
                                "Upload File(s)",
                                type="filepath",
                                file_count="multiple",
                                size="sm",
                            )
                            ingested_dataset = gr.List(
                                self._list_ingested_files,
                                headers=["File name"],
                                label="Ingested Files",
                                height=235,
                                interactive=False,
                                render=False,
                            )
                            upload_button.upload(
                                self._upload_file,
                                inputs=upload_button,
                                outputs=ingested_dataset,
                            ).then(
                                fn=lambda: (
                                    self._watcher_status_text(),
                                    self._watched_files_text(),
                                ),
                                outputs=[_watcher_status_cmp, _watcher_files_cmp],
                            )
                            ingested_dataset.change(
                                self._list_ingested_files,
                                outputs=ingested_dataset,
                            )
                            ingested_dataset.render()
                            deselect_file_button = gr.components.Button(
                                "De-select selected file",
                                size="sm",
                                interactive=False,
                            )
                            selected_text = gr.components.Textbox(
                                "All files",
                                label="Selected for Query or Deletion",
                                max_lines=1,
                            )
                            delete_file_button = gr.components.Button(
                                "🗑️ Delete selected file",
                                size="sm",
                                visible=settings().ui.delete_file_button_enabled,
                                interactive=False,
                            )
                            delete_files_button = gr.components.Button(
                                "⚠️ Delete ALL files",
                                size="sm",
                                visible=settings().ui.delete_all_files_button_enabled,
                            )
                            deselect_file_button.click(
                                self._deselect_selected_file,
                                outputs=[
                                    delete_file_button,
                                    deselect_file_button,
                                    selected_text,
                                ],
                            )
                            ingested_dataset.select(
                                fn=self._selected_a_file,
                                outputs=[
                                    delete_file_button,
                                    deselect_file_button,
                                    selected_text,
                                ],
                            )
                            delete_file_button.click(
                                self._delete_selected_file,
                                outputs=[
                                    ingested_dataset,
                                    delete_file_button,
                                    deselect_file_button,
                                    selected_text,
                                ],
                            )
                            delete_files_button.click(
                                self._delete_all_files,
                                outputs=[
                                    ingested_dataset,
                                    delete_file_button,
                                    deselect_file_button,
                                    selected_text,
                                ],
                            )

                            # ── Incremental mode info (read-only) ────
                            gr.Markdown("---")
                            gr.Markdown(
                                value=(
                                    "✅ **Incremental mode** — Only modified "
                                    "chunks are re-embedded on upload. "
                                    "Configure via `incremental.enabled` "
                                    "in settings.yaml."
                                    if self._incremental_enabled
                                    else "[i] **Standard mode** - Files are "
                                    "fully re-processed on every upload. "
                                    "Configure via `incremental.enabled` "
                                    "in settings.yaml."
                                ),
                            )

                            # ── File watcher control (incremental only) ──
                            if self._incremental_enabled:
                                gr.Markdown(
                                    "#### File Watcher\n"
                                    "Uploaded files are auto-registered for "
                                    "watching. Re-upload or edit the file in "
                                    "`local_data/uploads/` to trigger an "
                                    "incremental update. To watch a file "
                                    "in-place on your own disk (not via "
                                    "browser upload), paste its absolute path "
                                    "below."
                                )
                                _w_running = self._incremental_svc.is_watcher_running
                                _watcher_status_cmp.render()
                                with gr.Row():
                                    unwatch_all_btn = gr.Button(
                                        "Unwatch All Files",
                                        variant="stop",
                                        scale=1,
                                    )
                                with gr.Row():
                                    watcher_start_btn = gr.Button(
                                        "Start Watcher",
                                        variant="primary",
                                        interactive=not _w_running,
                                        scale=1,
                                    )
                                    watcher_stop_btn = gr.Button(
                                        "Stop Watcher",
                                        variant="stop",
                                        interactive=_w_running,
                                        scale=1,
                                    )
                                _watcher_files_cmp.render()

                                watch_path_input = gr.Textbox(
                                    label="Watch file by path (optional)",
                                    placeholder=(
                                        "Absolute path on this machine "
                                        "(e.g. C:\\docs\\report.md)"
                                    ),
                                    lines=1,
                                )
                                watch_path_btn = gr.Button(
                                    "Watch Path",
                                    variant="primary",
                                    scale=1,
                                )
                                watch_path_message = gr.Textbox(
                                    label="Result",
                                    interactive=False,
                                    lines=3,
                                )

                                _watcher_outputs = [
                                    _watcher_status_cmp,
                                    watcher_start_btn,
                                    watcher_stop_btn,
                                    _watcher_files_cmp,
                                ]
                                watcher_start_btn.click(
                                    fn=self._start_watcher,
                                    outputs=_watcher_outputs,
                                )
                                watcher_stop_btn.click(
                                    fn=self._stop_watcher,
                                    outputs=_watcher_outputs,
                                )
                                unwatch_all_btn.click(
                                    fn=self._unwatch_all,
                                    outputs=_watcher_outputs,
                                )
                                watch_path_btn.click(
                                    fn=self._watch_path,
                                    inputs=watch_path_input,
                                    outputs=[
                                        ingested_dataset,
                                        _watcher_status_cmp,
                                        _watcher_files_cmp,
                                        watch_path_message,
                                    ],
                                )

                            gr.Markdown("---")

                            system_prompt_input = gr.Textbox(
                                placeholder=self._system_prompt,
                                label="System Prompt",
                                lines=2,
                                interactive=True,
                                render=False,
                            )
                            mode.change(
                                self._set_current_mode,
                                inputs=mode,
                                outputs=[system_prompt_input, explanation_mode],
                            )
                            system_prompt_input.blur(
                                self._set_system_prompt,
                                inputs=system_prompt_input,
                            )

                        with gr.Column(scale=7, elem_id="col"):
                            _ = gr.ChatInterface(
                                self._chat,
                                chatbot=gr.Chatbot(
                                    label=label_text,
                                    show_copy_button=True,
                                    elem_id="chatbot",
                                    render=False,
                                    avatar_images=(None, AVATAR_BOT),
                                ),
                                additional_inputs=[
                                    mode,
                                    upload_button,
                                    system_prompt_input,
                                ],
                            )

                # ═══════════════ Tab 2: Benchmarks ══════════════════
                # Only mounted when ui.benchmarks_tab_enabled is true in
                # settings.yaml. Off by default: the benchmarks are for
                # development and thesis reproduction, not for end users.
                if settings().ui.benchmarks_tab_enabled:
                    with gr.Tab("Benchmarks"):
                        gr.Markdown(
                            "## Incremental RAG Benchmarks\n"
                            "Thesis evaluation benchmarks for the incremental "
                            "ingestion proof-of-concept. Results are shown as a "
                            "graph and raw console output.\n\n"
                            "| Button | What it measures |\n"
                            "|---|---|\n"
                            "| 🖥️ Computational Benchmark | **Real** wall-clock "
                            "embedding time: incremental (changed chunks only) vs "
                            "full re-embed. Requires `test_documents/` corpus. |\n"
                            "| 🧠 RAGAS Benchmark | Retrieval quality "
                            "(Context Recall, Context Precision) for semantic vs "
                            "fixed-size chunking. Pass `--ollama-url` for LLM "
                            "faithfulness evaluation. |\n"
                            "| ⚡ Incremental Speedtest | Chunk-level change "
                            "detection and diff efficiency. "
                            "**Note: embed times are estimated** — use "
                            "Computational Benchmark for real measurements. |\n"
                            "| 📈 Quality & Stability | Six-panel combined report: "
                            "incremental pipeline correctness (top row) and "
                            "full re-ingest baseline comparison (bottom row). |"
                        )

                        # ── Per-benchmark descriptions ────────────────
                        _DESC_COMPUTE = (
                            "### Computational Benchmark — Incremental vs Full Re-embed\n\n"
                            "Compares **real wall-clock `encode()` time** for incremental "
                            "(changed chunks only) vs full re-embed (all chunks, fixed-size "
                            "chunking — the PrivateGPT default).\n\n"
                            "| Element | Meaning |\n|---|---|\n"
                            "| X-axis | Change scenario: how much of the document was modified |\n"
                            "| Y-axis | Embedding time in seconds (only the `encode()` call) |\n"
                            "| Red bars | Full re-embed baseline (SentenceSplitter 1024 tokens) |\n"
                            "| Green bars | Incremental — only changed/added chunks re-embedded |\n\n"
                            "**How to read:** At 0 % change the green bar is near 0 s (nothing "
                            "to embed). As the change ratio rises both bars grow, but the green "
                            "bar grows slower. A green bar taller than red means the document is "
                            "small enough that chunking/diffing overhead outweighs the savings."
                        )
                        _DESC_RAGAS = (
                            "### RAGAS Benchmark — Retrieval Quality: Semantic vs Fixed-Size Chunking\n\n"
                            "Measures the **chunking strategy** dimension — independent of the "
                            "incremental update pipeline. Compares **semantic chunking** "
                            "(paragraph-boundary splits) vs **fixed-size chunking** "
                            "(LlamaIndex default) using the RAGAS evaluation framework on a "
                            "Golden Dataset.\n\n"
                            "| Element | Meaning |\n|---|---|\n"
                            "| X-axis | RAGAS metric |\n"
                            "| Bars | One colour per chunking strategy |\n"
                            "| Context Recall | Fraction of ground-truth information covered by "
                            "retrieved chunks — higher is fewer gaps |\n"
                            "| Context Precision | Fraction of retrieved content that is relevant "
                            "— higher is less noise |\n"
                            "| Faithfulness | Whether the LLM answer is grounded in the context "
                            "(requires `--ollama-url`, omitted otherwise) |\n\n"
                            "**How to read:** Semantic chunking should have the same or "
                            "higher recall because it keeps related sentences together. "
                            "Fixed-size 512 tokens often has higher precision (smaller, "
                            "focused chunks) but lower recall. Fixed-size 1024 tokens is "
                            "the PrivateGPT default."
                        )
                        _DESC_INCR = (
                            "### Incremental Speedtest — Chunk-Level Change Efficiency\n\n"
                            "Shows estimated speedup and embedding reuse ratio for different "
                            "change levels. **Note: embedding times are estimated** "
                            "(50 ms/chunk default) — use the Computational Benchmark for "
                            "real wall-clock measurements.\n\n"
                            "| Element | Meaning |\n|---|---|\n"
                            "| X-axis | Change scenario |\n"
                            "| Left Y-axis (line) | Speedup factor: how many times faster than "
                            "full re-embed |\n"
                            "| Right Y-axis (bars) | Efficiency %: fraction of chunks reused "
                            "(not re-embedded) |\n\n"
                            "**How to read:** At 0 % change efficiency is  100 % and speedup "
                            "is very high. At 50 % change about half the chunks are reused "
                            "(speedup  2x). At 90 % change most chunks changed and speedup is "
                            "low. A speedup of 1x means incremental and full take equal time."
                        )
                        _DESC_QUAL = (
                            "### Quality & Stability — Four-Panel Combined Report\n\n"
                            "**Top row — Incremental pipeline correctness:**\n\n"
                            "| Panel | What it shows |\n|---|---|\n"
                            "| Context Drift (left) | 0.0 = unchanged chunks kept "
                            "identical node IDs after every edit — no silent corruption |\n"
                            "| Pipeline Throughput (right) | Chunks/s for chunking + "
                            "diff detection combined (semantic). Drops at large sizes "
                            "because diff comparison is O(n²). |\n\n"
                            "**Bottom row — Full re-ingest baseline** (what happens "
                            "without the incremental pipeline):\n\n"
                            "| Panel | What it shows |\n|---|---|\n"
                            "| Full Re-ingest Waste (left) | Per sequential edit, the "
                            "fraction of chunks with identical hash that full re-ingest "
                            "re-embeds anyway. X-axis shows each edit step and its type "
                            "(mod/del/ins). **Semantic is higher** because stable "
                            "boundaries leave more unchanged chunks — more waste for "
                            "full re-ingest, bigger saving for incremental. |\n"
                            "| Avalanche Effect (right) | For one single-paragraph edit, "
                            "how many chunks change hash. Fixed-size: 25-100% (boundary "
                            "shift cascades). Semantic:  7% (only the edited chunk). |\n\n"
                            "**Note on 100% waste:** Delete operations produce 100% "
                            "waste for semantic — removing a paragraph leaves every "
                            "remaining chunk identical, so full re-ingest re-embeds "
                            "all of them for zero benefit."
                        )

                        with gr.Row():
                            btn_compute = gr.Button(
                                "Computational Benchmark", variant="primary"
                            )
                            btn_ragas = gr.Button("RAGAS Benchmark", variant="primary")
                            btn_incr = gr.Button(
                                "Incremental Speedtest", variant="primary"
                            )
                            btn_qual = gr.Button(
                                "Quality & Stability", variant="primary"
                            )

                        benchmark_description = gr.Markdown(
                            value="*Click a benchmark button above to see its description.*"
                        )
                        benchmark_plot = gr.Image(
                            label="Benchmark Graph",
                            type="filepath",
                            interactive=False,
                        )
                        benchmark_output = gr.Textbox(
                            label="Console Output",
                            lines=20,
                            max_lines=30,
                            interactive=False,
                            show_copy_button=True,
                        )

                        # Description appears immediately; plot+output follow when done
                        btn_compute.click(
                            fn=lambda: _DESC_COMPUTE,
                            outputs=benchmark_description,
                        ).then(
                            fn=functools.partial(
                                self._run_benchmark_with_plot,
                                "benchmark_compute",
                                ["--runs", "1"],
                            ),
                            outputs=[benchmark_plot, benchmark_output],
                        )
                        btn_ragas.click(
                            fn=lambda: _DESC_RAGAS,
                            outputs=benchmark_description,
                        ).then(
                            fn=functools.partial(
                                self._run_benchmark_with_plot, "benchmark_ragas"
                            ),
                            outputs=[benchmark_plot, benchmark_output],
                        )
                        btn_incr.click(
                            fn=lambda: _DESC_INCR,
                            outputs=benchmark_description,
                        ).then(
                            fn=functools.partial(
                                self._run_benchmark_with_plot,
                                "benchmark_incremental",
                            ),
                            outputs=[benchmark_plot, benchmark_output],
                        )
                        btn_qual.click(
                            fn=lambda: _DESC_QUAL,
                            outputs=benchmark_description,
                        ).then(
                            fn=functools.partial(
                                self._run_benchmark_with_plot, "benchmark_quality"
                            ),
                            outputs=[benchmark_plot, benchmark_output],
                        )

            with gr.Row():
                avatar_byte = AVATAR_BOT.read_bytes()
                f_base64 = f"data:image/png;base64,{base64.b64encode(avatar_byte).decode('utf-8')}"
                gr.HTML(
                    f"<div class='footer'><a class='footer-zylon-link' href='https://zylon.ai/'>Maintained by Zylon <img class='footer-zylon-ico' src='{f_base64}' alt=Zylon></a></div>"
                )

        return blocks

    def _run_benchmark_with_plot(
        self, script_name: str, script_args: list[str] | None = None
    ) -> tuple[str | None, str]:
        """Runs a benchmark script and generates a plot from its output."""
        output = ""
        plot_path = None
        try:
            # Ensure matplotlib is imported only when needed
            import sys
            import tempfile

            import matplotlib.pyplot as plt

            results_dir = Path(tempfile.mkdtemp())
            # Path(__file__) is in private_gpt/ui/ui.py
            # Move up two levels to get to project root, then scripts/
            script_path = (
                Path(__file__).resolve().parent.parent.parent
                / "scripts"
                / f"{script_name}.py"
            )

            command = [
                sys.executable,
                str(script_path),
                "--output-dir",
                str(results_dir),
            ]
            if script_args:
                command.extend(script_args)

            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                encoding="utf-8",
                errors="replace",
                env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
            )
            output = process.stdout + process.stderr

            if process.returncode != 0:
                output += (
                    f"\n\n⚠️ Benchmark script failed with exit code {process.returncode}"
                )
                return None, output

            if "compute" in script_name:
                csv_path = results_dir / "compute_benchmark.csv"
                if csv_path.exists():
                    plot_path = self._plot_compute(csv_path, plt, tempfile)

            elif "ragas" in script_name:
                csv_path = results_dir / "ragas_results.csv"
                if csv_path.exists():
                    plot_path = self._plot_ragas(csv_path, plt, tempfile)

            elif "incremental" in script_name:
                csv_path = results_dir / "benchmark_results.csv"
                if csv_path.exists():
                    plot_path = self._plot_incremental(csv_path, plt, tempfile)

            elif "quality" in script_name:
                # Run benchmark_quality_full into the same results_dir so both
                # CSVs land together, then generate a combined six-panel figure.
                full_script = (
                    Path(__file__).resolve().parent.parent.parent
                    / "scripts"
                    / "benchmark_quality_full.py"
                )
                subprocess.run(
                    [
                        sys.executable,
                        str(full_script),
                        "--output-dir",
                        str(results_dir),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    encoding="utf-8",
                    errors="replace",
                    env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
                )
                plot_path = self._plot_quality_combined(results_dir, plt, tempfile)

            return plot_path, output

        except ImportError:
            output += "\n\n[!] matplotlib not installed - no graph generated."
            return None, output
        except Exception as e:
            output += f"\n\n⚠️ Graph generation failed: {e}"
            return None, output

    # ── Matplotlib helpers ─────────────────────────────────────────

    @staticmethod
    def _plot_compute(csv_path: Path, plt: Any, tempfile: Any) -> str:
        """Bar chart: incremental vs full embedding time per scenario."""
        import csv as csv_mod

        scenarios, t_incr, t_full = [], [], []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                scenarios.append(row.get("label", row.get("Scenario", "")))
                t_incr.append(float(row.get("time_embed_incremental_s", "0")))
                t_full.append(float(row.get("time_embed_full_s", "0")))

        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(scenarios))
        w = 0.35
        ax.bar(
            [i - w / 2 for i in x],
            t_full,
            w,
            label="Full Re-ingest",
            color="#e74c3c",
            alpha=0.85,
        )
        ax.bar(
            [i + w / 2 for i in x],
            t_incr,
            w,
            label="Incremental",
            color="#2ecc71",
            alpha=0.85,
        )
        ax.set_xlabel("Scenario", fontsize=12)
        ax.set_ylabel("Embedding Time (seconds)", fontsize=12)
        ax.set_title(
            "Computational Benchmark: Incremental vs Full Re-ingest",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(scenarios, rotation=15, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=150)
        plt.close(fig)
        return str(tmp.name)

    @staticmethod
    def _plot_ragas(csv_path: Path, plt: Any, tempfile: Any) -> str:
        """Grouped bar chart for RAGAS metrics, color-coded by strategy."""
        import csv as csv_mod

        import numpy as np

        # Read data
        data = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                data.append(row)

        if not data:
            return ""  # Return empty string if no data

        # Extract unique strategies and prepare scores
        strategies = [row.get("strategy", row.get("Strategy", "")) for row in data]

        def _safe_float(val: str) -> float | None:
            try:
                return float(val)
            except (ValueError, TypeError):
                return None  # e.g. "N/A" when faithfulness was skipped

        all_metric_keys = ["context_recall", "context_precision", "faithfulness"]
        all_metric_labels = ["Context Recall", "Context Precision", "Faithfulness"]

        # Prepare scores for plotting — None when value is unavailable
        scores_by_strategy = {}
        for row in data:
            strategy_name = row.get("strategy", row.get("Strategy", ""))
            scores_by_strategy[strategy_name] = [
                _safe_float(row.get(key, "0")) for key in all_metric_keys
            ]

        # Drop any metric column where ALL strategies have None (e.g. faithfulness=N/A)
        keep = [
            i
            for i, _ in enumerate(all_metric_keys)
            if any(scores[i] is not None for scores in scores_by_strategy.values())
        ]
        metrics = [all_metric_labels[i] for i in keep]

        # Replace remaining None values with 0 for bar rendering
        for strategy_name in scores_by_strategy:
            full = scores_by_strategy[strategy_name]
            scores_by_strategy[strategy_name] = [
                (full[i] if full[i] is not None else 0.0) for i in keep
            ]

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(metrics))  # X-axis for metrics

        # Colors per strategy
        colors = [
            "#3498db",
            "#e67e22",
            "#9b59b6",
            "#2ecc71",
            "#f1c40f",
            "#e74c3c",
        ]  # More colors for more strategies

        num_strategies = len(strategies)
        if num_strategies > 0:
            w = 0.8 / num_strategies  # Width of each bar
            for i, strategy_name in enumerate(strategies):
                scores = scores_by_strategy.get(strategy_name, [0] * len(metrics))
                # Calculate x offsets for grouping
                offset = (i - num_strategies / 2.0 + 0.5) * w
                ax.bar(
                    x + offset,
                    scores,
                    w,
                    label=strategy_name,
                    color=colors[i % len(colors)],
                )

        ax.set_ylabel("Score (0-1)", fontsize=12)
        ax.set_title(
            "RAGAS Evaluation: Chunking Strategy Performance",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.legend(title="Strategy")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=150)
        plt.close(fig)
        return str(tmp.name)

    @staticmethod
    def _plot_incremental(csv_path: Path, plt: Any, tempfile: Any) -> str:
        """Line chart: speedup factor and efficiency % vs change level."""
        import csv as csv_mod

        labels, speedups, efficiencies = [], [], []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                label = row.get("experiment", row.get("Experiment", ""))
                # The no-change scenario reuses every embedding and produces
                # a speedup in the hundreds; it would flatten the y-axis for
                # all other scenarios. Drop it and disclose the exclusion.
                if "(0%)" in label or label.strip().lower() == "no change (0%)":
                    continue
                labels.append(label)
                sp = row.get("speedup_factor", "1")
                speedups.append(float(sp))
                eff = row.get("efficiency_ratio", "0")
                efficiencies.append(float(eff) * 100)

        fig, ax1 = plt.subplots(figsize=(10, 5))
        color1 = "#2980b9"
        ax1.set_xlabel("Change Scenario", fontsize=12)
        ax1.set_ylabel("Speedup Factor (x)", fontsize=12, color=color1)
        ax1.plot(
            labels,
            speedups,
            "o-",
            color=color1,
            linewidth=2,
            markersize=8,
            label="Speedup",
        )
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=20, ha="right")

        ax2 = ax1.twinx()
        color2 = "#27ae60"
        ax2.set_ylabel("Efficiency (%)", fontsize=12, color=color2)
        ax2.bar(
            range(len(labels)),
            efficiencies,
            alpha=0.3,
            color=color2,
            label="Efficiency %",
        )
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0, max(max(efficiencies, default=100) * 1.2, 110))

        title = "Incremental Pipeline: Speedup & Efficiency"
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.tight_layout()

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=150)
        plt.close(fig)
        return str(tmp.name)

    @staticmethod
    def _plot_quality_combined(results_dir: Path, plt: Any, tempfile: Any) -> str:
        """Four-panel combined Quality & Stability figure (2x2).

        Top row — incremental pipeline correctness:
          [0,0] Context drift per edit
          [0,1] Pipeline throughput (chunking + diff detection, semantic only)

        Bottom row — full re-ingest baseline:
          [1,0] Full re-ingest waste per iteration (by chunker, with edit labels)
          [1,1] Avalanche effect per single-edit scenario (by chunker)
        """
        import csv as csv_mod

        chunker_colors = {"fixed-size (1024/20)": "#e74c3c", "semantic": "#3498db"}

        # ── Load CSVs ──────────────────────────────────────────────────

        drift_labels, drift_scores = [], []
        drift_csv = results_dir / "quality_drift.csv"
        if drift_csv.exists():
            with open(drift_csv, encoding="utf-8") as f:
                for row in csv_mod.DictReader(f):
                    it = row.get("iteration", "?")
                    et = row.get("edit_type", "")[:3]
                    drift_labels.append(f"#{it} {et}")
                    drift_scores.append(float(row.get("drift_score", "0") or "0"))

        pipeline_x, pipeline_y = [], []
        pipeline_csv = results_dir / "quality_scalability.csv"
        if pipeline_csv.exists():
            with open(pipeline_csv, encoding="utf-8") as f:
                for row in csv_mod.DictReader(f):
                    try:
                        pipeline_x.append(int(row.get("num_paragraphs", 0)))
                        pipeline_y.append(float(row.get("chunks_per_second", 0) or "0"))
                    except ValueError:
                        pass

        # Inefficiency: store (iteration, ratio) per chunker + label per iteration
        ineff_by_chunker: dict[str, list[tuple[int, float]]] = {}
        ineff_labels: dict[int, str] = {}  # iteration -> "#N typ"
        ineff_csv = results_dir / "quality_full_inefficiency.csv"
        if ineff_csv.exists():
            with open(ineff_csv, encoding="utf-8") as f:
                for row in csv_mod.DictReader(f):
                    chunker = row.get("chunker", "?")
                    try:
                        it = int(row.get("iteration", "0") or "0")
                        ratio = float(row.get("inefficiency_ratio", "0") or "0")
                    except ValueError:
                        continue
                    ineff_by_chunker.setdefault(chunker, []).append((it, ratio))
                    if it not in ineff_labels:
                        et = row.get("edit_type", "")[:3]
                        ineff_labels[it] = f"#{it} {et}"

        scenarios: list[str] = []
        aval_by_chunker: dict[str, list[float]] = {}
        aval_csv = results_dir / "quality_full_avalanche.csv"
        if aval_csv.exists():
            with open(aval_csv, encoding="utf-8") as f:
                aval_rows = list(csv_mod.DictReader(f))
            for r in aval_rows:
                s = r.get("scenario", "?")
                if s not in scenarios:
                    scenarios.append(s)
            for r in aval_rows:
                c = r.get("chunker", "?")
                try:
                    ratio = float(r.get("avalanche_ratio", "0") or "0")
                except ValueError:
                    ratio = 0.0
                lst = aval_by_chunker.setdefault(c, [0.0] * len(scenarios))
                idx = scenarios.index(r.get("scenario", "?"))
                if idx < len(lst):
                    lst[idx] = ratio

        # ── Build 2x2 figure ──────────────────────────────────────────

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Quality & Stability — Incremental Pipeline (top) vs Full Re-ingest Baseline (bottom)",
            fontsize=13,
            fontweight="bold",
        )

        # ── [0,0]: Context Drift ───────────────────────────────────────
        ax = axes[0, 0]
        if drift_scores:
            if all(s == 0.0 for s in drift_scores):
                ax.scatter(
                    range(len(drift_labels)),
                    drift_scores,
                    color="#27ae60",
                    s=60,
                    zorder=5,
                )
                ax.axhline(0, color="#27ae60", linewidth=1.5, linestyle="--", alpha=0.6)
                ax.set_ylim(-0.05, 0.5)
                ax.text(
                    0.5,
                    0.55,
                    "All = 0.0\n(no node ID drift)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                    color="#1a7a4a",
                    bbox={
                        "boxstyle": "round,pad=0.4",
                        "facecolor": "#eafaf1",
                        "edgecolor": "#27ae60",
                        "alpha": 0.9,
                    },
                )
            else:
                ax.bar(
                    range(len(drift_labels)), drift_scores, color="#e74c3c", alpha=0.8
                )
                ax.set_ylim(0, max(max(drift_scores) * 1.5, 0.1))
            ax.set_xticks(range(len(drift_labels)))
            ax.set_xticklabels(drift_labels, rotation=45, ha="right", fontsize=7)
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
        ax.set_title("Context Drift per Edit", fontsize=11, fontweight="bold")
        ax.set_xlabel("Cumulative edit (iter type)", fontsize=9)
        ax.set_ylabel("Drift score (0 = stable)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # ── [0,1]: Pipeline throughput ─────────────────────────────────
        ax = axes[0, 1]
        if pipeline_x and pipeline_y:
            ax.plot(
                pipeline_x,
                pipeline_y,
                marker="o",
                color="#3498db",
                linewidth=2,
                markersize=6,
            )
            ax.fill_between(pipeline_x, pipeline_y, alpha=0.15, color="#3498db")
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
        ax.set_title(
            "Pipeline Throughput\n(chunking + diff detection, semantic)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Document size (paragraphs)", fontsize=9)
        ax.set_ylabel("Chunks / second", fontsize=9)
        ax.grid(axis="both", alpha=0.3)

        # ── [1,0]: Full re-ingest waste with per-step edit labels ──────
        ax = axes[1, 0]
        if ineff_by_chunker:
            sorted_iters = sorted(ineff_labels.keys())
            x_labels = [ineff_labels[i] for i in sorted_iters]
            for chunker, series in ineff_by_chunker.items():
                series.sort(key=lambda s: s[0])
                ax.plot(
                    range(len(series)),
                    [s[1] * 100 for s in series],
                    marker="o",
                    linewidth=2,
                    color=chunker_colors.get(chunker, "#7f8c8d"),
                    label=chunker,
                )
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
            ax.legend(fontsize=9)
            ax.set_ylim(0, 105)
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
        ax.set_title(
            "Full Re-ingest Waste per Iteration\n(higher = more savings from incremental)",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_xlabel("Edit step (iter type)", fontsize=9)
        ax.set_ylabel("Chunks re-embedded despite\nidentical hash (%)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # ── [1,1]: Avalanche effect ────────────────────────────────────
        ax = axes[1, 1]
        if scenarios and aval_by_chunker:
            import numpy as np

            x = np.arange(len(scenarios))
            num = len(aval_by_chunker)
            w = 0.8 / max(num, 1)
            for i, (chunker, vals) in enumerate(aval_by_chunker.items()):
                offset = (i - num / 2.0 + 0.5) * w
                ax.bar(
                    x + offset,
                    [v * 100 for v in vals],
                    w,
                    color=chunker_colors.get(chunker, "#7f8c8d"),
                    label=chunker,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(scenarios, rotation=28, ha="right", fontsize=8)
            ax.legend(fontsize=9)
            ax.set_ylim(0, 105)
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
        ax.set_title(
            "Avalanche Effect\n(single-paragraph edit)", fontsize=11, fontweight="bold"
        )
        ax.set_xlabel("Edit scenario", fontsize=9)
        ax.set_ylabel("Chunks with changed hash (%)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        fig.savefig(tmp.name, dpi=130, bbox_inches="tight")
        plt.close(fig)
        return str(tmp.name)

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        logger.info("Mounting the gradio UI, at path=%s", path)
        gr.mount_gradio_app(app, blocks, path=path, favicon_path=AVATAR_BOT)


if __name__ == "__main__":
    ui = global_injector.get(PrivateGptUi)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)
