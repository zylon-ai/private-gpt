"""PrivateGPT Chainlit UI.

Replaces the Gradio UI with a more customizable Chainlit interface.
Supports all chat modes (RAG, Search, Basic, Summarize), file management,
optional authentication, and streaming responses.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import chainlit as cl
from chainlit.input_widget import Select, TextInput
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole

from private_gpt.di import global_injector
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.recipes.summarize.summarize_service import SummarizeService
from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)

# ── Mode descriptions ──────────────────────────────────────────────────────────

_MODE_DESCRIPTIONS = {
    "RAG": "Answer questions using your ingested documents as context. Sources are cited.",
    "Search": "Search for the most relevant passages across all ingested documents.",
    "Basic": "Chat directly with the AI without any document context.",
    "Summarize": "Generate a summary of a selected document. Select a file first.",
}

_DEFAULT_SYSTEM_PROMPTS = {
    "RAG": settings.ui.default_query_system_prompt,
    "Search": "",
    "Basic": settings.ui.default_chat_system_prompt,
    "Summarize": settings.ui.default_summarization_system_prompt,
}

# ── Auth (conditional) ─────────────────────────────────────────────────────────

if settings.user_auth.enabled:
    from private_gpt.server.auth.user_store import UserStore

    @cl.password_auth_callback
    def auth_callback(username: str, password: str) -> Optional[cl.User]:
        try:
            user_store = global_injector.get(UserStore)
            if user_store.verify_password(username, password):
                record = user_store.get_user(username)
                collections = user_store.get_user_collections(username)
                return cl.User(
                    identifier=username,
                    metadata={
                        "is_admin": record.is_admin if record else False,
                        "accessible_collections": collections,
                    },
                )
        except Exception:
            logger.warning("Auth failed for user=%s", username, exc_info=True)
        return None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_chat_settings(mode: str) -> cl.ChatSettings:
    default_modes = ["RAG", "Search", "Basic", "Summarize"]
    return cl.ChatSettings(
        [
            Select(
                id="mode",
                label="Chat Mode",
                values=default_modes,
                initial_value=mode,
                description="How the AI processes your message",
            ),
            TextInput(
                id="system_prompt",
                label="System Prompt",
                initial_value=_DEFAULT_SYSTEM_PROMPTS.get(mode, ""),
                description="Customize the AI's behavior and persona",
                multiline=True,
            ),
        ]
    )


def _build_context_filter(selected_file: Optional[str]) -> Optional[ContextFilter]:
    if not selected_file:
        return None
    ingest_svc = global_injector.get(IngestService)
    all_docs = ingest_svc.list_ingested()
    doc_ids = [
        doc.doc_id
        for doc in all_docs
        if doc.doc_metadata and doc.doc_metadata.get("file_name") == selected_file
    ]
    return ContextFilter(docs_ids=doc_ids) if doc_ids else None


async def _stream_sync_gen(sync_gen, msg: cl.Message) -> None:
    """Stream tokens from a synchronous generator to a Chainlit message."""
    loop = asyncio.get_event_loop()
    sentinel = object()

    def get_next() -> object:
        return next(sync_gen, sentinel)  # type: ignore[call-overload]

    while True:
        item = await loop.run_in_executor(None, get_next)
        if item is sentinel:
            break
        if isinstance(item, str):
            await msg.stream_token(item)
        elif isinstance(item, ChatResponse) and item.delta:
            await msg.stream_token(item.delta)


# ── Session lifecycle ──────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start() -> None:
    user = cl.context.session.user
    username = user.identifier if user else "anonymous"
    is_admin = (
        user.metadata.get("is_admin", False) if user and user.metadata else False
    )
    accessible_collections = (
        user.metadata.get("accessible_collections", []) if user and user.metadata else []
    )

    default_mode = settings.ui.default_mode or "RAG"

    cl.user_session.set("username", username)
    cl.user_session.set("is_admin", is_admin)
    cl.user_session.set("accessible_collections", accessible_collections)
    cl.user_session.set("mode", default_mode)
    cl.user_session.set("system_prompt", _DEFAULT_SYSTEM_PROMPTS.get(default_mode, ""))
    cl.user_session.set("selected_file", None)
    cl.user_session.set("chat_history", [])

    await _build_chat_settings(default_mode).send()

    actions = [
        cl.Action(
            name="list_files",
            value="list",
            label="📁 Manage Files",
            description="Show, select, or delete ingested files",
        ),
    ]
    if is_admin:
        actions.append(
            cl.Action(
                name="admin_panel",
                value="admin",
                label="⚙ Admin Panel",
                description="Manage users, groups, and collections",
            )
        )

    mode_desc = _MODE_DESCRIPTIONS.get(default_mode, "")
    await cl.Message(
        content=(
            f"**Welcome to PrivateGPT!**\n\n"
            f"**Mode:** {default_mode} — {mode_desc}\n\n"
            f"Use the ⚙ **Settings** panel to change mode and system prompt.\n"
            f"Attach files to your message to upload and ingest them."
        ),
        actions=actions,
    ).send()


@cl.on_settings_update
async def on_settings_update(updated: dict) -> None:  # type: ignore[type-arg]
    new_mode = updated.get("mode", "RAG")
    new_prompt = updated.get("system_prompt", _DEFAULT_SYSTEM_PROMPTS.get(new_mode, ""))
    cl.user_session.set("mode", new_mode)
    cl.user_session.set("system_prompt", new_prompt)
    cl.user_session.set("chat_history", [])  # Reset history on mode change
    mode_desc = _MODE_DESCRIPTIONS.get(new_mode, "")
    await cl.Message(
        content=f"Mode changed to **{new_mode}** — {mode_desc}"
    ).send()


# ── File management actions ────────────────────────────────────────────────────

@cl.action_callback("list_files")
async def on_list_files(action: cl.Action) -> None:
    ingest_svc = global_injector.get(IngestService)
    docs = ingest_svc.list_ingested()

    if not docs:
        await cl.Message(
            content=(
                "No files ingested yet.\n\n"
                "Attach files to your message using the 📎 button to upload them."
            )
        ).send()
        return

    # Deduplicate by file name
    seen_files: dict[str, str] = {}  # file_name -> doc_id (first occurrence)
    for doc in docs:
        fname = (
            doc.doc_metadata.get("file_name", doc.doc_id) if doc.doc_metadata else doc.doc_id
        )
        if fname not in seen_files:
            seen_files[fname] = doc.doc_id

    selected = cl.user_session.get("selected_file")
    lines = []
    actions = []
    for fname, doc_id in seen_files.items():
        marker = " ✓ (selected)" if fname == selected else ""
        lines.append(f"• **{fname}**{marker}")
        actions.append(
            cl.Action(
                name="select_file",
                value=fname,
                label=f"Select: {fname[:35]}",
                description=f"Use {fname} as context for RAG/Summarize",
            )
        )
        actions.append(
            cl.Action(
                name="delete_file",
                value=fname,
                label=f"Delete: {fname[:30]}",
                description=f"Remove {fname} from the index",
            )
        )

    if selected:
        actions.append(
            cl.Action(
                name="deselect_file",
                value="",
                label="Clear selection",
                description="Use all documents instead of a specific file",
            )
        )

    content = "**Ingested Files:**\n" + "\n".join(lines)
    await cl.Message(content=content, actions=actions).send()


@cl.action_callback("select_file")
async def on_select_file(action: cl.Action) -> None:
    fname = action.value
    cl.user_session.set("selected_file", fname)
    await cl.Message(
        content=f"Selected **{fname}** as context for RAG and Summarize modes."
    ).send()


@cl.action_callback("deselect_file")
async def on_deselect_file(action: cl.Action) -> None:
    cl.user_session.set("selected_file", None)
    await cl.Message(content="File selection cleared. All documents will be used.").send()


@cl.action_callback("delete_file")
async def on_delete_file(action: cl.Action) -> None:
    fname = action.value
    ingest_svc = global_injector.get(IngestService)
    docs = ingest_svc.list_ingested()

    deleted = 0
    for doc in docs:
        doc_fname = (
            doc.doc_metadata.get("file_name", "") if doc.doc_metadata else ""
        )
        if doc_fname == fname:
            ingest_svc.delete(doc.doc_id)
            deleted += 1

    if cl.user_session.get("selected_file") == fname:
        cl.user_session.set("selected_file", None)

    await cl.Message(
        content=f"Deleted **{fname}** ({deleted} chunk(s) removed)."
    ).send()


@cl.action_callback("admin_panel")
async def on_admin_panel(action: cl.Action) -> None:
    await cl.Message(
        content=(
            "**Admin Panel**\n\n"
            "Manage users, groups, and collections via the REST API:\n\n"
            "| Endpoint | Method | Description |\n"
            "|----------|--------|-------------|\n"
            "| `/v1/auth/users` | POST/DELETE | Create or delete users |\n"
            "| `/v1/auth/groups` | POST/DELETE | Create or delete groups |\n"
            "| `/v1/auth/collections` | POST/DELETE | Create or delete collections |\n"
            "| `/v1/auth/users/{user}/groups` | PUT | Assign user to group |\n"
            "| `/v1/auth/groups/{group}/collections` | PUT | Assign group to collection |\n\n"
            "Or use the interactive API docs at `/docs`."
        )
    ).send()


# ── File upload handling ───────────────────────────────────────────────────────

async def _handle_file_uploads(elements: list) -> list[str]:  # type: ignore[type-arg]
    ingest_svc = global_injector.get(IngestService)
    ingested_names: list[str] = []

    selected_collection: Optional[str] = cl.user_session.get("selected_collection")

    for element in elements:
        if not hasattr(element, "path") or not element.path:
            continue
        fname = element.name or Path(element.path).name
        try:
            await asyncio.to_thread(
                ingest_svc.ingest_file,
                fname,
                Path(element.path),
                selected_collection,
            )
            ingested_names.append(fname)
        except Exception:
            logger.error("Failed to ingest file=%s", fname, exc_info=True)

    if ingested_names:
        await cl.Message(
            content=f"Ingested {len(ingested_names)} file(s): {', '.join(ingested_names)}\n\nUse **Manage Files** to select a file for RAG or Summarize."
        ).send()

    return ingested_names


# ── Message handler ────────────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message) -> None:
    # Handle file attachments first
    if message.elements:
        await _handle_file_uploads(message.elements)
        if not message.content.strip():
            return

    mode: str = cl.user_session.get("mode", "RAG")
    system_prompt: str = cl.user_session.get("system_prompt", "")
    selected_file: Optional[str] = cl.user_session.get("selected_file")
    chat_history: list[tuple[str, str]] = cl.user_session.get("chat_history", [])

    if mode == "RAG":
        response_text = await _handle_rag(
            message.content, system_prompt, selected_file, chat_history
        )
    elif mode == "Search":
        response_text = await _handle_search(message.content, selected_file)
    elif mode == "Basic":
        response_text = await _handle_basic(
            message.content, system_prompt, chat_history
        )
    elif mode == "Summarize":
        response_text = await _handle_summarize(
            message.content, selected_file, system_prompt
        )
    else:
        response_text = f"Unknown mode: {mode}"
        await cl.Message(content=response_text).send()
        return

    # Update chat history (only for conversational modes)
    if mode in ("RAG", "Basic"):
        chat_history.append((message.content, response_text))
        cl.user_session.set("chat_history", chat_history[-20:])  # Keep last 20 turns


# ── Chat modes ─────────────────────────────────────────────────────────────────

async def _handle_rag(
    query: str,
    system_prompt: str,
    selected_file: Optional[str],
    history: list[tuple[str, str]],
) -> str:
    chat_svc = global_injector.get(ChatService)
    context_filter = _build_context_filter(selected_file)

    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))
    for user_msg, assistant_msg in history:
        messages.append(ChatMessage(role=MessageRole.USER, content=user_msg))
        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=assistant_msg))
    messages.append(ChatMessage(role=MessageRole.USER, content=query))

    msg = cl.Message(content="")
    await msg.send()
    full_response = ""

    try:
        completion_gen: CompletionGen = await asyncio.to_thread(
            chat_svc.stream_chat,
            messages,
            True,
            context_filter,
        )
        await _stream_sync_gen(iter(completion_gen.response), msg)

        if completion_gen.sources:
            sources_text = "\n\n---\n**Sources:**\n"
            seen: set[str] = set()
            for i, chunk in enumerate(completion_gen.sources, 1):
                meta = chunk.document.doc_metadata or {}
                fname = meta.get("file_name", "-")
                page = meta.get("page_label", "-")
                key = f"{fname}-{page}"
                if key not in seen:
                    sources_text += f"{i}. {fname} (page {page})\n"
                    seen.add(key)
            await msg.stream_token(sources_text)

        full_response = msg.content
        await msg.update()
    except Exception:
        logger.error("RAG error", exc_info=True)
        await msg.update()

    return full_response


async def _handle_search(query: str, selected_file: Optional[str]) -> str:
    chunks_svc = global_injector.get(ChunksService)
    context_filter = _build_context_filter(selected_file)

    try:
        chunks: list[Chunk] = await asyncio.to_thread(
            chunks_svc.retrieve_relevant,
            query,
            context_filter,
            10,
            0,
        )
    except Exception:
        logger.error("Search error", exc_info=True)
        await cl.Message(content="Error during search.").send()
        return ""

    if not chunks:
        await cl.Message(content="No relevant passages found.").send()
        return ""

    lines = [f"**Search results for:** '{query}'\n"]
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.document.doc_metadata or {}
        fname = meta.get("file_name", "-")
        page = meta.get("page_label", "-")
        preview = chunk.text[:400] + "..." if len(chunk.text) > 400 else chunk.text
        lines.append(f"**{i}. {fname}** (page {page}, score {chunk.score:.3f})\n{preview}\n")

    content = "\n".join(lines)
    await cl.Message(content=content).send()
    return content


async def _handle_basic(
    query: str,
    system_prompt: str,
    history: list[tuple[str, str]],
) -> str:
    chat_svc = global_injector.get(ChatService)

    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage(role=MessageRole.SYSTEM, content=system_prompt))
    for user_msg, assistant_msg in history:
        messages.append(ChatMessage(role=MessageRole.USER, content=user_msg))
        messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=assistant_msg))
    messages.append(ChatMessage(role=MessageRole.USER, content=query))

    msg = cl.Message(content="")
    await msg.send()

    try:
        completion_gen: CompletionGen = await asyncio.to_thread(
            chat_svc.stream_chat,
            messages,
            False,
            None,
        )
        await _stream_sync_gen(iter(completion_gen.response), msg)
        await msg.update()
    except Exception:
        logger.error("Basic chat error", exc_info=True)
        await msg.update()

    return msg.content


async def _handle_summarize(
    query: str,
    selected_file: Optional[str],
    system_prompt: str,
) -> str:
    if not selected_file:
        await cl.Message(
            content=(
                "Please select a file first.\n\n"
                "Click **Manage Files** to choose a document to summarize."
            )
        ).send()
        return ""

    summarize_svc = global_injector.get(SummarizeService)
    context_filter = _build_context_filter(selected_file)

    instructions = query.strip() if query.strip() else None
    prompt = system_prompt if system_prompt.strip() else None

    msg = cl.Message(content="")
    await msg.send()

    try:
        token_gen = await asyncio.to_thread(
            summarize_svc.stream_summarize,
            True,
            None,
            instructions,
            context_filter,
            prompt,
        )
        await _stream_sync_gen(iter(token_gen), msg)
        await msg.update()
    except Exception:
        logger.error("Summarize error", exc_info=True)
        msg.content = "Error during summarization. Make sure documents are ingested and a file is selected."
        await msg.update()

    return msg.content
