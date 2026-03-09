"""PrivateGPT Gradio UI — with optional user/group authentication.

When ``user_auth.enabled`` is False (default) the UI behaves exactly as before:
no login screen, all documents visible.

When ``user_auth.enabled`` is True:
- A login tab is presented first.
- After login the user sees only collections they have access to.
- Admins see an additional "Admin" tab for managing users, groups and collections.
- Per-session state (selected file, system prompt, auth) is stored in ``gr.State``
  so multiple browser sessions do not share state.
"""

import base64
import logging
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
from private_gpt.server.auth.collection_service import CollectionService
from private_gpt.server.auth.token_service import TokenService
from private_gpt.server.auth.user_store import UserStore
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.recipes.summarize.summarize_service import SummarizeService
from private_gpt.settings.settings import settings
from private_gpt.ui.images import logo_svg

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
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


# ── Default session state ─────────────────────────────────────────────────────

def _empty_session() -> dict:
    """Return a fresh, unauthenticated session state dict."""
    return {
        "username": None,
        "token": None,
        "is_admin": False,
        "accessible_collections": [],
        "selected_collection": None,
        "selected_filename": None,
        "system_prompt": "",
    }


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
            curated_sources = list(dict.fromkeys(curated_sources).keys())
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
        user_store: UserStore,
        token_service: TokenService,
        collection_service: CollectionService,
    ) -> None:
        self._ingest_service = ingest_service
        self._chat_service = chat_service
        self._chunks_service = chunks_service
        self._summarize_service = summarizeService
        self._user_store = user_store
        self._token_service = token_service
        self._collection_service = collection_service

        self._ui_block = None

        default_mode_map = {mode.value: mode for mode in Modes}
        self._default_mode = default_mode_map.get(
            settings().ui.default_mode, Modes.RAG_MODE
        )
        self._auth_enabled = settings().user_auth.enabled

    # ── Auth helpers ──────────────────────────────────────────────────────────

    def _do_login(
        self, username: str, password: str, state: dict
    ) -> tuple[dict, str, Any, Any, Any, Any]:
        """Validate credentials and update session state.

        Returns: (new_state, status_msg, login_col_update, main_col_update,
                  collection_dd_update, user_badge_update)
        """
        if not username or not password:
            return state, "⚠️ Please enter username and password.", gr.update(), gr.update(), gr.update(), gr.update()

        if not self._user_store.verify_password(username, password):
            return state, "❌ Invalid username or password.", gr.update(), gr.update(), gr.update(), gr.update()

        token = self._token_service.create_token(username)
        user = self._user_store.get_user(username)
        collections = self._user_store.get_user_collections(username)

        new_state = {
            "username": username,
            "token": token,
            "is_admin": user.is_admin if user else False,
            "accessible_collections": collections,
            "selected_collection": collections[0] if collections else None,
            "selected_filename": None,
            "system_prompt": self._get_default_system_prompt(self._default_mode),
        }

        # Show collection dropdown with accessible collections
        col_choices = collections if collections else []
        col_value = col_choices[0] if col_choices else None
        col_update = gr.update(
            choices=col_choices, value=col_value, visible=True
        )

        user_badge_update = gr.update(
            value=f"👤 **{username}**{'  🔑 Admin' if new_state['is_admin'] else ''}",
            visible=True,
        )

        return (
            new_state,
            f"✅ Welcome, {username}!",
            gr.update(visible=False),   # hide login column
            gr.update(visible=True),    # show main content column
            col_update,
            user_badge_update,
        )

    def _do_logout(self, state: dict) -> tuple[dict, Any, Any, Any, Any]:
        """Clear session and show login screen."""
        return (
            _empty_session(),
            gr.update(visible=True),   # show login column
            gr.update(visible=False),  # hide main content column
            gr.update(value="", visible=False),  # hide user badge
            gr.update(choices=[], value=None, visible=False),  # hide collection dropdown
        )

    # ── Document helpers ──────────────────────────────────────────────────────

    def _list_ingested_files(self, state: dict | None = None) -> list[list[str]]:
        """Return [[file_name], ...] filtered by current collection if auth is on."""
        files: set[str] = set()
        collection = state.get("selected_collection") if state else None

        for doc in self._ingest_service.list_ingested():
            if doc.doc_metadata is None:
                continue
            # When auth is enabled, filter by active collection
            if self._auth_enabled and collection is not None:
                if doc.doc_metadata.get("collection_name") != collection:
                    continue
            file_name = doc.doc_metadata.get("file_name", "[FILE NAME MISSING]")
            files.add(file_name)
        return [[row] for row in sorted(files)]

    def _upload_file(self, files: list[str], state: dict) -> list[list[str]]:
        logger.debug("Loading count=%s files", len(files))
        paths = [Path(f) for f in files]
        collection = state.get("selected_collection") if self._auth_enabled else None

        file_names = [path.name for path in paths]
        doc_ids_to_delete = []
        for doc in self._ingest_service.list_ingested():
            if doc.doc_metadata and doc.doc_metadata.get("file_name") in file_names:
                # Only replace within same collection when auth is enabled
                if self._auth_enabled and collection is not None:
                    if doc.doc_metadata.get("collection_name") != collection:
                        continue
                doc_ids_to_delete.append(doc.doc_id)

        for doc_id in doc_ids_to_delete:
            self._ingest_service.delete(doc_id)

        self._ingest_service.bulk_ingest(
            [(str(path.name), path) for path in paths],
            collection_name=collection,
        )
        return self._list_ingested_files(state)

    def _delete_selected_file(self, state: dict) -> tuple[list[list[str]], Any, Any, Any, dict]:
        selected = state.get("selected_filename")
        logger.debug("Deleting selected file=%s", selected)
        for doc in self._ingest_service.list_ingested():
            if doc.doc_metadata and doc.doc_metadata.get("file_name") == selected:
                self._ingest_service.delete(doc.doc_id)
        new_state = {**state, "selected_filename": None}
        return (
            self._list_ingested_files(new_state),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value="All files"),
            new_state,
        )

    def _delete_all_files(self, state: dict) -> tuple[list[list[str]], Any, Any, Any, dict]:
        collection = state.get("selected_collection") if self._auth_enabled else None
        for doc in self._ingest_service.list_ingested():
            if self._auth_enabled and collection is not None:
                if not doc.doc_metadata or doc.doc_metadata.get("collection_name") != collection:
                    continue
            self._ingest_service.delete(doc.doc_id)
        new_state = {**state, "selected_filename": None}
        return (
            self._list_ingested_files(new_state),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value="All files"),
            new_state,
        )

    def _selected_a_file(
        self, select_data: gr.SelectData, state: dict
    ) -> tuple[Any, Any, Any, dict]:
        new_state = {**state, "selected_filename": select_data.value}
        return (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(value=select_data.value),
            new_state,
        )

    def _deselect_selected_file(self, state: dict) -> tuple[Any, Any, Any, dict]:
        new_state = {**state, "selected_filename": None}
        return (
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(value="All files"),
            new_state,
        )

    def _on_collection_change(
        self, collection_name: str, state: dict
    ) -> tuple[dict, list[list[str]]]:
        new_state = {**state, "selected_collection": collection_name, "selected_filename": None}
        return new_state, self._list_ingested_files(new_state)

    # ── Chat / RAG helpers ────────────────────────────────────────────────────

    @staticmethod
    def _get_default_system_prompt(mode: Modes) -> str:
        match mode:
            case Modes.RAG_MODE:
                return settings().ui.default_query_system_prompt or ""
            case Modes.BASIC_CHAT_MODE:
                return settings().ui.default_chat_system_prompt or ""
            case Modes.SUMMARIZE_MODE:
                return settings().ui.default_summarization_system_prompt or ""
            case _:
                return ""

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

    def _set_current_mode(self, mode: Modes, state: dict) -> tuple[Any, Any, dict]:
        new_prompt = self._get_default_system_prompt(mode)
        new_state = {**state, "system_prompt": new_prompt}
        interactive = bool(new_prompt)
        return (
            gr.update(placeholder=new_prompt, interactive=interactive),
            gr.update(value=self._get_default_mode_explanation(mode)),
            new_state,
        )

    def _set_system_prompt(self, system_prompt_input: str, state: dict) -> dict:
        logger.info("Setting system prompt to: %s", system_prompt_input)
        return {**state, "system_prompt": system_prompt_input}

    def _build_context_filter(self, state: dict) -> ContextFilter | None:
        """Build a ContextFilter based on selected file and/or active collection."""
        selected_file = state.get("selected_filename")
        selected_collection = state.get("selected_collection") if self._auth_enabled else None

        docs_ids: list[str] | None = None

        all_docs = self._ingest_service.list_ingested()

        if selected_file is not None:
            # Filter to the specific file (within collection if auth enabled)
            docs_ids = [
                doc.doc_id
                for doc in all_docs
                if doc.doc_metadata
                and doc.doc_metadata.get("file_name") == selected_file
                and (
                    not self._auth_enabled
                    or selected_collection is None
                    or doc.doc_metadata.get("collection_name") == selected_collection
                )
            ]
        elif self._auth_enabled and selected_collection is not None:
            # Filter to entire collection
            docs_ids = [
                doc.doc_id
                for doc in all_docs
                if doc.doc_metadata
                and doc.doc_metadata.get("collection_name") == selected_collection
            ]

        if docs_ids is not None:
            return ContextFilter(docs_ids=docs_ids)
        return None

    def _chat(
        self,
        message: str,
        history: list[list[str]],
        mode: Modes,
        state: dict,
        *_: Any,
    ) -> Any:
        def yield_deltas(completion_gen: CompletionGen) -> Iterable[str]:
            full_response: str = ""
            for delta in completion_gen.response:
                if isinstance(delta, str):
                    full_response += delta
                elif isinstance(delta, ChatResponse):
                    full_response += delta.delta or ""
                yield full_response
                time.sleep(0.02)

            if completion_gen.sources:
                full_response += SOURCES_SEPARATOR
                cur_sources = Source.curate_sources(completion_gen.sources)
                sources_text = "\n\n\n"
                used_files: set[str] = set()
                for index, source in enumerate(cur_sources, start=1):
                    key = f"{source.file}-{source.page}"
                    if key not in used_files:
                        sources_text += f"{index}. {source.file} (page {source.page}) \n\n"
                        used_files.add(key)
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
                            content=interaction[1].split(SOURCES_SEPARATOR)[0],
                            role=MessageRole.ASSISTANT,
                        )
                    )
            return history_messages[:20]

        system_prompt = state.get("system_prompt", "")
        new_message = ChatMessage(content=message, role=MessageRole.USER)
        all_messages = [*build_history(), new_message]
        if system_prompt:
            all_messages.insert(
                0, ChatMessage(content=system_prompt, role=MessageRole.SYSTEM)
            )

        context_filter = self._build_context_filter(state)

        match mode:
            case Modes.RAG_MODE:
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
                    f"{index}. **{source.file} (page {source.page})**\n {source.text}"
                    for index, source in enumerate(sources, start=1)
                )

            case Modes.SUMMARIZE_MODE:
                summary_stream = self._summarize_service.stream_summarize(
                    use_context=True,
                    context_filter=context_filter,
                    instructions=message,
                )
                yield from yield_tokens(summary_stream)

    # ── Admin panel helpers ───────────────────────────────────────────────────

    def _admin_list_users(self) -> list[list[str]]:
        users = self._user_store.list_users()
        return [
            [u.username, "Yes" if u.is_admin else "No", ", ".join(u.groups)]
            for u in users
        ]

    def _admin_create_user(
        self, username: str, password: str, is_admin: bool
    ) -> tuple[str, list[list[str]]]:
        username = username.strip()
        if not username or not password:
            return "⚠️ Username and password are required.", self._admin_list_users()
        try:
            self._user_store.create_user(username, password, is_admin=is_admin)
            return f"✅ User '{username}' created.", self._admin_list_users()
        except Exception as e:
            return f"❌ Error: {e}", self._admin_list_users()

    def _admin_delete_user(
        self, username: str, current_state: dict
    ) -> tuple[str, list[list[str]]]:
        username = username.strip()
        if username == current_state.get("username"):
            return "❌ Cannot delete yourself.", self._admin_list_users()
        try:
            self._user_store.delete_user(username)
            return f"✅ User '{username}' deleted.", self._admin_list_users()
        except Exception as e:
            return f"❌ Error: {e}", self._admin_list_users()

    def _admin_list_groups(self) -> list[list[str]]:
        groups = self._user_store.list_groups()
        return [
            [g.group_name, ", ".join(g.collections)] for g in groups
        ]

    def _admin_create_group(self, group_name: str) -> tuple[str, list[list[str]]]:
        group_name = group_name.strip()
        if not group_name:
            return "⚠️ Group name is required.", self._admin_list_groups()
        try:
            self._user_store.create_group(group_name)
            return f"✅ Group '{group_name}' created.", self._admin_list_groups()
        except Exception as e:
            return f"❌ Error: {e}", self._admin_list_groups()

    def _admin_delete_group(self, group_name: str) -> tuple[str, list[list[str]]]:
        group_name = group_name.strip()
        try:
            self._user_store.delete_group(group_name)
            return f"✅ Group '{group_name}' deleted.", self._admin_list_groups()
        except Exception as e:
            return f"❌ Error: {e}", self._admin_list_groups()

    def _admin_assign_user_group(
        self, username: str, group_name: str
    ) -> tuple[str, list[list[str]]]:
        try:
            self._user_store.assign_user_to_group(username.strip(), group_name.strip())
            return f"✅ User '{username}' added to group '{group_name}'.", self._admin_list_users()
        except Exception as e:
            return f"❌ Error: {e}", self._admin_list_users()

    def _admin_assign_group_collection(
        self, group_name: str, collection_name: str
    ) -> tuple[str, list[list[str]]]:
        try:
            self._user_store.assign_collection_to_group(
                group_name.strip(), collection_name.strip()
            )
            return (
                f"✅ Collection '{collection_name}' assigned to group '{group_name}'.",
                self._admin_list_groups(),
            )
        except Exception as e:
            return f"❌ Error: {e}", self._admin_list_groups()

    def _admin_list_collections(self) -> list[list[str]]:
        cols = self._user_store.list_collections()
        return [[c.collection_name, c.display_name] for c in cols]

    def _admin_create_collection(
        self, collection_name: str, display_name: str
    ) -> tuple[str, list[list[str]]]:
        collection_name = collection_name.strip()
        if not collection_name:
            return "⚠️ Collection name is required.", self._admin_list_collections()
        try:
            self._user_store.create_collection(collection_name, display_name.strip())
            return (
                f"✅ Collection '{collection_name}' created.",
                self._admin_list_collections(),
            )
        except Exception as e:
            return f"❌ Error: {e}", self._admin_list_collections()

    def _admin_delete_collection(
        self, collection_name: str
    ) -> tuple[str, list[list[str]]]:
        try:
            self._user_store.delete_collection(collection_name.strip())
            return (
                f"✅ Collection '{collection_name}' deleted.",
                self._admin_list_collections(),
            )
        except Exception as e:
            return f"❌ Error: {e}", self._admin_list_collections()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")

        css = (
            ".logo { display:flex; background-color: #C7BAFF; height: 80px; "
            "border-radius: 8px; align-content: center; justify-content: center; "
            "align-items: center; }"
            ".logo img { height: 25% }"
            ".contain { display: flex !important; flex-direction: column !important; }"
            "#component-0, #component-3, #component-10, #component-8 { height: 100% !important; }"
            "#chatbot { flex-grow: 1 !important; overflow: auto !important; }"
            "#col { height: calc(100vh - 112px - 16px) !important; }"
            "hr { margin-top: 1em; margin-bottom: 1em; border: 0; border-top: 1px solid #FFF; }"
            ".avatar-image { background-color: antiquewhite; border-radius: 2px; }"
            ".footer { text-align: center; margin-top: 20px; font-size: 14px; "
            "display: flex; align-items: center; justify-content: center; }"
            ".footer-zylon-link { display:flex; margin-left: 5px; text-decoration: auto; "
            "color: var(--body-text-color); }"
            ".footer-zylon-link:hover { color: #C7BAFF; }"
            ".footer-zylon-ico { height: 20px; margin-left: 5px; "
            "background-color: antiquewhite; border-radius: 2px; }"
        )

        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft(primary_hue=slate),
            css=css,
        ) as blocks:
            # Per-session state
            session_state = gr.State(_empty_session())

            # ── Header ────────────────────────────────────────────────────────
            with gr.Row():
                with gr.Column(scale=8):
                    gr.HTML(
                        f"<div class='logo'><img src={logo_svg} alt=PrivateGPT></div>"
                    )
                with gr.Column(scale=2):
                    user_badge = gr.Markdown("", visible=False)
                    logout_button = gr.Button(
                        "Logout",
                        size="sm",
                        visible=False,
                    )

            # ── Login panel (only visible when auth is enabled and user is logged out) ─
            with gr.Column(visible=self._auth_enabled) as login_col:
                gr.Markdown("## Sign in to PrivateGPT")
                login_username = gr.Textbox(label="Username", placeholder="username")
                login_password = gr.Textbox(
                    label="Password", type="password", placeholder="••••••••"
                )
                login_button = gr.Button("Sign in", variant="primary")
                login_status = gr.Markdown("")

            # ── Main content (hidden until logged in, or always visible if auth off) ──
            with gr.Column(visible=not self._auth_enabled) as main_col:
                with gr.Tabs():
                    # ── Chat Tab ──────────────────────────────────────────────
                    with gr.Tab("💬 Chat"):
                        with gr.Row(equal_height=False):
                            # Left sidebar
                            with gr.Column(scale=3):
                                default_mode = self._default_mode

                                # Collection selector (only in auth mode)
                                collection_selector = gr.Dropdown(
                                    label="Active Collection",
                                    choices=[],
                                    interactive=True,
                                    visible=False,
                                )

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
                                upload_button = gr.components.UploadButton(
                                    "Upload File(s)",
                                    type="filepath",
                                    file_count="multiple",
                                    size="sm",
                                )
                                ingested_dataset = gr.List(
                                    lambda: self._list_ingested_files(None),
                                    headers=["File name"],
                                    label="Ingested Files",
                                    height=235,
                                    interactive=False,
                                    render=False,
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

                                system_prompt_input = gr.Textbox(
                                    placeholder=self._get_default_system_prompt(
                                        default_mode
                                    ),
                                    label="System Prompt",
                                    lines=2,
                                    interactive=True,
                                    render=False,
                                )

                                def get_model_label() -> str | None:
                                    cfg = settings()
                                    if cfg is None:
                                        return None
                                    llm_mode = cfg.llm.mode
                                    model_mapping = {
                                        "llamacpp": cfg.llamacpp.llm_hf_model_file,
                                        "openai": cfg.openai.model,
                                        "openailike": cfg.openai.model,
                                        "azopenai": cfg.azopenai.llm_model,
                                        "sagemaker": cfg.sagemaker.llm_endpoint_name,
                                        "mock": llm_mode,
                                        "ollama": cfg.ollama.llm_model,
                                        "gemini": cfg.gemini.model,
                                    }
                                    return model_mapping.get(llm_mode)

                            # Right: chat panel
                            with gr.Column(scale=7, elem_id="col"):
                                model_label = get_model_label()
                                label_text = (
                                    f"LLM: {settings().llm.mode} | Model: {model_label}"
                                    if model_label
                                    else f"LLM: {settings().llm.mode}"
                                )
                                system_prompt_input.render()
                                _ = gr.ChatInterface(
                                    self._chat,
                                    chatbot=gr.Chatbot(
                                        label=label_text,
                                        show_copy_button=True,
                                        elem_id="chatbot",
                                        render=False,
                                        avatar_images=(None, AVATAR_BOT),
                                    ),
                                    additional_inputs=[mode, session_state, upload_button, system_prompt_input],
                                )

                    # ── Admin Tab (visible only when user is admin) ───────────
                    with gr.Tab("⚙️ Admin") as admin_tab:
                        admin_notice = gr.Markdown(
                            "🔒 *Admin access required. Please log in as an admin.*"
                            if self._auth_enabled
                            else ""
                        )
                        with gr.Column(visible=not self._auth_enabled) as admin_content:
                            with gr.Tabs():
                                # Users sub-tab
                                with gr.Tab("Users"):
                                    user_table = gr.Dataframe(
                                        headers=["Username", "Admin", "Groups"],
                                        value=self._admin_list_users(),
                                        interactive=False,
                                        label="Users",
                                    )
                                    with gr.Row():
                                        new_username = gr.Textbox(
                                            label="Username", placeholder="new_user"
                                        )
                                        new_password = gr.Textbox(
                                            label="Password",
                                            type="password",
                                            placeholder="••••••••",
                                        )
                                        new_is_admin = gr.Checkbox(
                                            label="Admin", value=False
                                        )
                                    create_user_btn = gr.Button(
                                        "Create User", variant="primary", size="sm"
                                    )
                                    with gr.Row():
                                        del_username = gr.Textbox(
                                            label="Username to delete",
                                            placeholder="username",
                                        )
                                    delete_user_btn = gr.Button(
                                        "Delete User", variant="stop", size="sm"
                                    )
                                    with gr.Row():
                                        assign_u = gr.Textbox(
                                            label="Username", placeholder="username"
                                        )
                                        assign_g = gr.Textbox(
                                            label="Group", placeholder="group_name"
                                        )
                                    assign_user_group_btn = gr.Button(
                                        "Assign User to Group", size="sm"
                                    )
                                    user_action_status = gr.Markdown("")

                                # Groups sub-tab
                                with gr.Tab("Groups"):
                                    group_table = gr.Dataframe(
                                        headers=["Group Name", "Collections"],
                                        value=self._admin_list_groups(),
                                        interactive=False,
                                        label="Groups",
                                    )
                                    new_group_name = gr.Textbox(
                                        label="Group name", placeholder="eng-team"
                                    )
                                    create_group_btn = gr.Button(
                                        "Create Group", variant="primary", size="sm"
                                    )
                                    del_group_name = gr.Textbox(
                                        label="Group to delete", placeholder="group_name"
                                    )
                                    delete_group_btn = gr.Button(
                                        "Delete Group", variant="stop", size="sm"
                                    )
                                    with gr.Row():
                                        gc_group = gr.Textbox(
                                            label="Group", placeholder="group_name"
                                        )
                                        gc_collection = gr.Textbox(
                                            label="Collection",
                                            placeholder="collection_name",
                                        )
                                    assign_gc_btn = gr.Button(
                                        "Assign Collection to Group", size="sm"
                                    )
                                    group_action_status = gr.Markdown("")

                                # Collections sub-tab
                                with gr.Tab("Collections"):
                                    collection_table = gr.Dataframe(
                                        headers=["Collection Name", "Display Name"],
                                        value=self._admin_list_collections(),
                                        interactive=False,
                                        label="Collections",
                                    )
                                    with gr.Row():
                                        new_col_name = gr.Textbox(
                                            label="Collection name",
                                            placeholder="engineering",
                                        )
                                        new_col_display = gr.Textbox(
                                            label="Display name",
                                            placeholder="Engineering Docs",
                                        )
                                    create_col_btn = gr.Button(
                                        "Create Collection", variant="primary", size="sm"
                                    )
                                    del_col_name = gr.Textbox(
                                        label="Collection to delete",
                                        placeholder="collection_name",
                                    )
                                    delete_col_btn = gr.Button(
                                        "Delete Collection", variant="stop", size="sm"
                                    )
                                    col_action_status = gr.Markdown("")

            # ── Footer ────────────────────────────────────────────────────────
            with gr.Row():
                avatar_byte = AVATAR_BOT.read_bytes()
                f_base64 = f"data:image/png;base64,{base64.b64encode(avatar_byte).decode('utf-8')}"
                gr.HTML(
                    f"<div class='footer'>"
                    f"<a class='footer-zylon-link' href='https://zylon.ai/'>"
                    f"Maintained by Zylon "
                    f"<img class='footer-zylon-ico' src='{f_base64}' alt=Zylon>"
                    f"</a></div>"
                )

            # ── Event bindings ────────────────────────────────────────────────

            # Login
            login_button.click(
                self._do_login,
                inputs=[login_username, login_password, session_state],
                outputs=[
                    session_state,
                    login_status,
                    login_col,
                    main_col,
                    collection_selector,
                    user_badge,
                ],
            )
            login_password.submit(
                self._do_login,
                inputs=[login_username, login_password, session_state],
                outputs=[
                    session_state,
                    login_status,
                    login_col,
                    main_col,
                    collection_selector,
                    user_badge,
                ],
            )

            # After login, show logout button, conditionally show admin content
            def _post_login_ui(state: dict) -> tuple[Any, Any, Any]:
                logout_vis = state.get("username") is not None
                admin_vis = state.get("is_admin", False) or not self._auth_enabled
                return (
                    gr.update(visible=logout_vis),
                    gr.update(visible=admin_vis),
                    gr.update(value="" if admin_vis else "🔒 *Admin access required.*"),
                )

            login_button.click(
                _post_login_ui,
                inputs=[session_state],
                outputs=[logout_button, admin_content, admin_notice],
            )
            login_password.submit(
                _post_login_ui,
                inputs=[session_state],
                outputs=[logout_button, admin_content, admin_notice],
            )

            # Logout
            logout_button.click(
                self._do_logout,
                inputs=[session_state],
                outputs=[
                    session_state,
                    login_col,
                    main_col,
                    user_badge,
                    collection_selector,
                ],
            )
            logout_button.click(
                lambda: (gr.update(visible=False), gr.update(visible=False)),
                outputs=[logout_button, admin_content],
            )

            # Collection selector
            collection_selector.change(
                self._on_collection_change,
                inputs=[collection_selector, session_state],
                outputs=[session_state, ingested_dataset],
            )

            # Mode change
            mode.change(
                self._set_current_mode,
                inputs=[mode, session_state],
                outputs=[system_prompt_input, explanation_mode, session_state],
            )

            # System prompt
            system_prompt_input.blur(
                self._set_system_prompt,
                inputs=[system_prompt_input, session_state],
                outputs=[session_state],
            )

            # File upload
            upload_button.upload(
                self._upload_file,
                inputs=[upload_button, session_state],
                outputs=[ingested_dataset],
            )

            # File selection
            ingested_dataset.select(
                self._selected_a_file,
                inputs=[session_state],
                outputs=[delete_file_button, deselect_file_button, selected_text, session_state],
            )
            ingested_dataset.change(
                self._list_ingested_files,
                inputs=[session_state],
                outputs=[ingested_dataset],
            )

            # File de-selection
            deselect_file_button.click(
                self._deselect_selected_file,
                inputs=[session_state],
                outputs=[delete_file_button, deselect_file_button, selected_text, session_state],
            )

            # Delete selected
            delete_file_button.click(
                self._delete_selected_file,
                inputs=[session_state],
                outputs=[ingested_dataset, delete_file_button, deselect_file_button, selected_text, session_state],
            )

            # Delete all
            delete_files_button.click(
                self._delete_all_files,
                inputs=[session_state],
                outputs=[ingested_dataset, delete_file_button, deselect_file_button, selected_text, session_state],
            )

            # ── Admin bindings ────────────────────────────────────────────────

            create_user_btn.click(
                self._admin_create_user,
                inputs=[new_username, new_password, new_is_admin],
                outputs=[user_action_status, user_table],
            )
            delete_user_btn.click(
                self._admin_delete_user,
                inputs=[del_username, session_state],
                outputs=[user_action_status, user_table],
            )
            assign_user_group_btn.click(
                self._admin_assign_user_group,
                inputs=[assign_u, assign_g],
                outputs=[user_action_status, user_table],
            )
            create_group_btn.click(
                self._admin_create_group,
                inputs=[new_group_name],
                outputs=[group_action_status, group_table],
            )
            delete_group_btn.click(
                self._admin_delete_group,
                inputs=[del_group_name],
                outputs=[group_action_status, group_table],
            )
            assign_gc_btn.click(
                self._admin_assign_group_collection,
                inputs=[gc_group, gc_collection],
                outputs=[group_action_status, group_table],
            )
            create_col_btn.click(
                self._admin_create_collection,
                inputs=[new_col_name, new_col_display],
                outputs=[col_action_status, collection_table],
            )
            delete_col_btn.click(
                self._admin_delete_collection,
                inputs=[del_col_name],
                outputs=[col_action_status, collection_table],
            )

        return blocks

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
