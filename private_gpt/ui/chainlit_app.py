"""PrivateGPT Chainlit UI.

Replaces the Gradio UI with a more customizable Chainlit interface.
Supports all chat modes (RAG, Search, Basic, Summarize), file management,
optional authentication, and streaming responses.

Navigation is exposed as dedicated Chat Profiles in the Chainlit sidebar:
  • Chat  – normal RAG / Search / Basic / Summarize conversation
  • Files – upload, list, select and delete ingested documents
  • Admin – user, group and collection management (admin-only)
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
from private_gpt.settings.settings import unsafe_typed_settings as settings

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


# ── Chat Profiles (sidebar navigation) ────────────────────────────────────────

@cl.set_chat_profiles
async def chat_profiles(current_user: Optional[cl.User] = None) -> list[cl.ChatProfile]:
    """Expose Files and Admin as dedicated sidebar menu entries."""
    # When user_auth is disabled everyone is treated as admin for the UI.
    is_admin = (
        current_user.metadata.get("is_admin", False)
        if current_user and current_user.metadata
        else not settings.user_auth.enabled
    )

    profiles: list[cl.ChatProfile] = [
        cl.ChatProfile(
            name="Chat",
            markdown_description=(
                "**Chat** — Use RAG, Search, Basic or Summarize modes "
                "to interact with your documents."
            ),
        ),
        cl.ChatProfile(
            name="Files",
            markdown_description=(
                "**Files** — Upload, browse, select or delete ingested documents."
            ),
        ),
    ]

    if is_admin:
        profiles.append(
            cl.ChatProfile(
                name="Admin",
                markdown_description=(
                    "**Admin** — Manage users, groups and collections "
                    "(administrator access required)."
                ),
            )
        )

    return profiles


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


# ── Admin wizard helper ────────────────────────────────────────────────────────

async def _admin_flow_continue(user_input: str) -> None:
    """Advance the current multi-step admin wizard with the user's latest input."""
    flow: dict | None = cl.user_session.get("admin_flow")
    if not flow:
        return

    flow_type = flow["type"]
    step = flow["step"]

    if settings.user_auth.enabled:
        from private_gpt.server.auth.user_store import UserStore as _UserStore
        user_store = global_injector.get(_UserStore)
    else:
        await cl.Message(content="User authentication is disabled.").send()
        cl.user_session.set("admin_flow", None)
        return

    # ── create_user wizard ─────────────────────────────────────────────────
    if flow_type == "create_user":
        if step == "username":
            if not user_input:
                await cl.Message(content="Username cannot be empty. Operation cancelled.").send()
                cl.user_session.set("admin_flow", None)
                return
            flow["username"] = user_input
            flow["step"] = "password"
            cl.user_session.set("admin_flow", flow)
            res = await cl.AskUserMessage(
                content=f"Enter a **password** for **{user_input}**:", timeout=120
            ).send()
            if res is None:
                cl.user_session.set("admin_flow", None)
                return
            await _admin_flow_continue(res["output"].strip())

        elif step == "password":
            if not user_input:
                await cl.Message(content="Password cannot be empty. Operation cancelled.").send()
                cl.user_session.set("admin_flow", None)
                return
            flow["password"] = user_input
            flow["step"] = "is_admin"
            cl.user_session.set("admin_flow", flow)
            res = await cl.AskUserMessage(
                content="Should this user be an **admin**? Type `yes` or `no`:", timeout=120
            ).send()
            if res is None:
                cl.user_session.set("admin_flow", None)
                return
            await _admin_flow_continue(res["output"].strip().lower())

        elif step == "is_admin":
            is_admin_flag = user_input in ("yes", "y", "true", "1")
            try:
                user_store.create_user(flow["username"], flow["password"], is_admin=is_admin_flag)
                role = "admin" if is_admin_flag else "regular user"
                await cl.Message(content=f"Created {role} **{flow['username']}**.").send()
            except Exception:
                logger.error("Failed to create user=%s", flow["username"], exc_info=True)
                await cl.Message(content="Failed to create user.").send()
            cl.user_session.set("admin_flow", None)
            await _show_admin_panel()

    # ── create_group wizard ────────────────────────────────────────────────
    elif flow_type == "create_group":
        if step == "group_name":
            if not user_input:
                await cl.Message(content="Group name cannot be empty. Operation cancelled.").send()
                cl.user_session.set("admin_flow", None)
                return
            try:
                user_store.create_group(user_input)
                await cl.Message(content=f"Created group **{user_input}**.").send()
            except Exception:
                logger.error("Failed to create group=%s", user_input, exc_info=True)
                await cl.Message(content="Failed to create group.").send()
            cl.user_session.set("admin_flow", None)
            await _show_admin_panel()

    # ── create_collection wizard ───────────────────────────────────────────
    elif flow_type == "create_collection":
        if step == "collection_name":
            if not user_input:
                await cl.Message(content="Collection name cannot be empty. Operation cancelled.").send()
                cl.user_session.set("admin_flow", None)
                return
            flow["collection_name"] = user_input
            flow["step"] = "display_name"
            cl.user_session.set("admin_flow", flow)
            res = await cl.AskUserMessage(
                content=(
                    f"Enter a **display name** for collection `{user_input}` "
                    "(or send an empty message to use the collection name):"
                ),
                timeout=120,
            ).send()
            if res is None:
                cl.user_session.set("admin_flow", None)
                return
            await _admin_flow_continue(res["output"].strip())

        elif step == "display_name":
            col_name = flow["collection_name"]
            display = user_input if user_input else col_name
            try:
                user_store.create_collection(col_name, display_name=display)
                await cl.Message(
                    content=f"Created collection **{col_name}** (displayed as *{display}*)."
                ).send()
            except Exception:
                logger.error("Failed to create collection=%s", col_name, exc_info=True)
                await cl.Message(content="Failed to create collection.").send()
            cl.user_session.set("admin_flow", None)
            await _show_admin_panel()

    # ── add_user_to_group wizard ───────────────────────────────────────────
    elif flow_type == "add_user_to_group":
        if step == "group_name":
            if not user_input:
                await cl.Message(content="Group name cannot be empty. Operation cancelled.").send()
                cl.user_session.set("admin_flow", None)
                return
            username = flow["username"]
            try:
                user_store.assign_user_to_group(username, user_input)
                await cl.Message(content=f"Added **{username}** to group **{user_input}**.").send()
            except Exception:
                logger.error("Failed to add user=%s to group=%s", username, user_input, exc_info=True)
                await cl.Message(content="Operation failed.").send()
            cl.user_session.set("admin_flow", None)
            await _show_admin_panel()

    # ── add_collection_to_group wizard ────────────────────────────────────
    elif flow_type == "add_collection_to_group":
        if step == "collection_name":
            if not user_input:
                await cl.Message(content="Collection name cannot be empty. Operation cancelled.").send()
                cl.user_session.set("admin_flow", None)
                return
            group_name = flow["group_name"]
            try:
                user_store.assign_collection_to_group(group_name, user_input)
                await cl.Message(
                    content=f"Assigned collection **{user_input}** to group **{group_name}**."
                ).send()
            except Exception:
                logger.error(
                    "Failed to assign col=%s to group=%s", user_input, group_name, exc_info=True
                )
                await cl.Message(content="Operation failed.").send()
            cl.user_session.set("admin_flow", None)
            await _show_admin_panel()


# ── File manager helper ────────────────────────────────────────────────────────

async def _show_file_manager() -> None:
    """Display the file management panel with collection filter and per-file actions."""
    selected_collection: Optional[str] = cl.user_session.get("selected_collection")
    accessible_collections: list[str] = cl.user_session.get("accessible_collections") or []

    # Show collection filter buttons when the user has named collections
    if settings.user_auth.enabled and accessible_collections:
        filter_actions = [
            cl.Action(
                name="select_collection",
                value=col,
                label=col,
                description=f"Show only documents in {col}",
                payload={"value": col},
            )
            for col in accessible_collections
        ] + [
            cl.Action(
                name="select_collection",
                value="__all__",
                label="All Documents",
                description="Show documents from all collections",
                payload={"value": "__all__"},
            )
        ]
        current_label = selected_collection or "All Documents"
        await cl.Message(
            content=f"**Collection filter** — currently showing: *{current_label}*",
            actions=filter_actions,
        ).send()

    ingest_svc = global_injector.get(IngestService)
    docs = ingest_svc.list_ingested()

    # Filter by selected collection
    if selected_collection:
        docs = [
            d
            for d in docs
            if d.doc_metadata and d.doc_metadata.get("collection_name") == selected_collection
        ]

    if not docs:
        no_docs_msg = (
            f"No files ingested in collection **{selected_collection}**."
            if selected_collection
            else "**No files ingested yet.**"
        )
        await cl.Message(
            content=f"{no_docs_msg}\n\nAttach files to your message using the 📎 button to upload them."
        ).send()
        return

    # Deduplicate by file name; keep (doc_id, collection_name)
    seen_files: dict[str, tuple[str, Optional[str]]] = {}
    for doc in docs:
        fname = (
            doc.doc_metadata.get("file_name", doc.doc_id) if doc.doc_metadata else doc.doc_id
        )
        col = doc.doc_metadata.get("collection_name") if doc.doc_metadata else None
        if fname not in seen_files:
            seen_files[fname] = (doc.doc_id, col)

    selected_file = cl.user_session.get("selected_file")
    lines = ["**Ingested Files:**\n"]
    actions = []
    for fname, (_doc_id, col) in seen_files.items():
        file_marker = " ✓ *(selected)*" if fname == selected_file else ""
        col_tag = f"  *(collection: {col})*" if col else ""
        lines.append(f"• **{fname}**{file_marker}{col_tag}")
        actions.append(
            cl.Action(
                name="select_file",
                value=fname,
                label=f"Select: {fname[:35]}",
                description=f"Use {fname} as context for RAG/Summarize",
                payload={"value": fname},
            )
        )
        actions.append(
            cl.Action(
                name="delete_file",
                value=fname,
                label=f"Delete: {fname[:30]}",
                description=f"Remove {fname} from the index",
                payload={"value": fname},
            )
        )

    if selected_file:
        actions.append(
            cl.Action(
                name="deselect_file",
                value="",
                label="Clear selection",
                description="Use all documents instead of a specific file",
                payload={"value": ""},
            )
        )

    await cl.Message(content="\n".join(lines), actions=actions).send()


# ── Admin panel helper ─────────────────────────────────────────────────────────

async def _show_admin_panel() -> None:
    """Display the interactive admin panel with live user/group/collection management."""
    await cl.Message(content="## Admin Panel\n\nManage users, groups, and collections below.").send()

    if not settings.user_auth.enabled:
        await cl.Message(
            content=(
                "User authentication is **disabled**. "
                "Enable `user_auth.enabled` in settings to manage users, groups, and collections."
            )
        ).send()
        return

    from private_gpt.server.auth.user_store import UserStore as _UserStore
    user_store = global_injector.get(_UserStore)
    current_username = cl.user_session.get("username")

    # ── Users section ──────────────────────────────────────────────────────────
    users = user_store.list_users()
    user_lines = ["### Users\n"]
    user_actions: list[cl.Action] = [
        cl.Action(
            name="admin_create_user_start",
            value="",
            label="+ Create User",
            description="Create a new user account",
            payload={},
        )
    ]
    if not users:
        user_lines.append("*(no users)*")
    for u in users:
        role = " *(admin)*" if u.is_admin else ""
        groups_str = ", ".join(u.groups) if u.groups else "*(none)*"
        user_lines.append(f"• **{u.username}**{role} — groups: *{groups_str}*")
        # Delete button (cannot delete self)
        if u.username != current_username:
            user_actions.append(
                cl.Action(
                    name="admin_delete_user",
                    value=u.username,
                    label=f"Delete {u.username}",
                    description=f"Permanently delete user {u.username}",
                    payload={"username": u.username},
                )
            )
        # Add to group button
        user_actions.append(
            cl.Action(
                name="admin_add_user_to_group_start",
                value=u.username,
                label=f"Add {u.username[:20]} to Group",
                description=f"Add {u.username} to a group",
                payload={"username": u.username},
            )
        )
        # Remove from group buttons
        for g in u.groups:
            user_actions.append(
                cl.Action(
                    name="admin_remove_user_from_group",
                    value=f"{u.username}|{g}",
                    label=f"Remove {u.username[:15]} from {g[:15]}",
                    description=f"Remove {u.username} from group {g}",
                    payload={"username": u.username, "group_name": g},
                )
            )
    await cl.Message(content="\n".join(user_lines), actions=user_actions).send()

    # ── Groups section ─────────────────────────────────────────────────────────
    groups = user_store.list_groups()
    group_lines = ["### Groups\n"]
    group_actions: list[cl.Action] = [
        cl.Action(
            name="admin_create_group_start",
            value="",
            label="+ Create Group",
            description="Create a new group",
            payload={},
        )
    ]
    if not groups:
        group_lines.append("*(no groups)*")
    for g in groups:
        cols_str = ", ".join(g.collections) if g.collections else "*(none)*"
        group_lines.append(f"• **{g.group_name}** — collections: *{cols_str}*")
        group_actions.append(
            cl.Action(
                name="admin_delete_group",
                value=g.group_name,
                label=f"Delete {g.group_name}",
                description=f"Delete group {g.group_name}",
                payload={"group_name": g.group_name},
            )
        )
        group_actions.append(
            cl.Action(
                name="admin_add_collection_to_group_start",
                value=g.group_name,
                label=f"Add Collection to {g.group_name[:20]}",
                description=f"Assign a collection to group {g.group_name}",
                payload={"group_name": g.group_name},
            )
        )
        for col in g.collections:
            group_actions.append(
                cl.Action(
                    name="admin_remove_collection_from_group",
                    value=f"{g.group_name}|{col}",
                    label=f"Remove {col[:15]} from {g.group_name[:15]}",
                    description=f"Remove collection {col} from group {g.group_name}",
                    payload={"group_name": g.group_name, "collection_name": col},
                )
            )
    await cl.Message(content="\n".join(group_lines), actions=group_actions).send()

    # ── Collections section ────────────────────────────────────────────────────
    collections = user_store.list_collections()
    col_lines = ["### Collections\n"]
    col_actions: list[cl.Action] = [
        cl.Action(
            name="admin_create_collection_start",
            value="",
            label="+ Create Collection",
            description="Create a new document collection",
            payload={},
        )
    ]
    if not collections:
        col_lines.append("*(no collections)*")
    for c in collections:
        display = c.display_name if c.display_name != c.collection_name else ""
        display_str = f" ({display})" if display else ""
        col_lines.append(f"• **{c.collection_name}**{display_str}")
        col_actions.append(
            cl.Action(
                name="admin_delete_collection",
                value=c.collection_name,
                label=f"Delete {c.collection_name}",
                description=f"Delete collection {c.collection_name}",
                payload={"collection_name": c.collection_name},
            )
        )
    await cl.Message(content="\n".join(col_lines), actions=col_actions).send()


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

    # Determine active profile (set by set_chat_profiles selector)
    chat_profile: str = cl.user_session.get("chat_profile") or "Chat"

    default_mode = settings.ui.default_mode or "RAG"

    cl.user_session.set("username", username)
    cl.user_session.set("is_admin", is_admin)
    cl.user_session.set("accessible_collections", accessible_collections)
    cl.user_session.set("mode", default_mode)
    cl.user_session.set("system_prompt", _DEFAULT_SYSTEM_PROMPTS.get(default_mode, ""))
    cl.user_session.set("selected_file", None)
    cl.user_session.set("selected_collection", None)
    cl.user_session.set("admin_flow", None)
    cl.user_session.set("chat_history", [])

    if chat_profile == "Files":
        await cl.Message(
            content=(
                "## 📁 File Manager\n\n"
                "Here you can browse, select, and delete your ingested documents.\n"
                "Attach files to your message using the 📎 button to upload new ones.\n\n"
                "Loading your files…"
            )
        ).send()
        await _show_file_manager()

    elif chat_profile == "Admin":
        if not is_admin and settings.user_auth.enabled:
            await cl.Message(
                content="⛔ **Access denied.** Administrator privileges are required."
            ).send()
        else:
            await _show_admin_panel()

    else:
        # Default "Chat" profile
        await _build_chat_settings(default_mode).send()

        mode_desc = _MODE_DESCRIPTIONS.get(default_mode, "")
        await cl.Message(
            content=(
                f"**Welcome to PrivateGPT!**\n\n"
                f"**Mode:** {default_mode} — {mode_desc}\n\n"
                f"Use the ⚙ **Settings** panel to change mode and system prompt.\n"
                f"Switch to the **Files** tab in the sidebar to manage documents.\n"
                f"Attach files to your message to upload and ingest them."
            )
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


# ── File management action callbacks ──────────────────────────────────────────

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


# ── Collection selection callback ──────────────────────────────────────────────

@cl.action_callback("select_collection")
async def on_select_collection(action: cl.Action) -> None:
    payload = action.payload or {}
    value = payload.get("value") or action.value
    if value == "__all__":
        cl.user_session.set("selected_collection", None)
        await cl.Message(content="Showing **all documents** across all collections.").send()
    else:
        cl.user_session.set("selected_collection", value)
        await cl.Message(content=f"Filtering documents to collection **{value}**.").send()
    await _show_file_manager()


# ── Admin: simple destructive action callbacks ─────────────────────────────────

@cl.action_callback("admin_delete_user")
async def on_admin_delete_user(action: cl.Action) -> None:
    payload = action.payload or {}
    username = payload.get("username") or action.value
    current = cl.user_session.get("username")
    if username == current:
        await cl.Message(content="You cannot delete your own account.").send()
        return
    if not settings.user_auth.enabled:
        return
    from private_gpt.server.auth.user_store import UserStore as _UserStore
    try:
        global_injector.get(_UserStore).delete_user(username)
        await cl.Message(content=f"User **{username}** deleted.").send()
    except Exception:
        logger.error("Failed to delete user=%s", username, exc_info=True)
        await cl.Message(content=f"Failed to delete user **{username}**.").send()
    await _show_admin_panel()


@cl.action_callback("admin_delete_group")
async def on_admin_delete_group(action: cl.Action) -> None:
    payload = action.payload or {}
    group_name = payload.get("group_name") or action.value
    if not settings.user_auth.enabled:
        return
    from private_gpt.server.auth.user_store import UserStore as _UserStore
    try:
        global_injector.get(_UserStore).delete_group(group_name)
        await cl.Message(content=f"Group **{group_name}** deleted.").send()
    except Exception:
        logger.error("Failed to delete group=%s", group_name, exc_info=True)
        await cl.Message(content=f"Failed to delete group **{group_name}**.").send()
    await _show_admin_panel()


@cl.action_callback("admin_delete_collection")
async def on_admin_delete_collection(action: cl.Action) -> None:
    payload = action.payload or {}
    col_name = payload.get("collection_name") or action.value
    if not settings.user_auth.enabled:
        return
    from private_gpt.server.auth.user_store import UserStore as _UserStore
    try:
        global_injector.get(_UserStore).delete_collection(col_name)
        await cl.Message(content=f"Collection **{col_name}** deleted.").send()
    except Exception:
        logger.error("Failed to delete collection=%s", col_name, exc_info=True)
        await cl.Message(content=f"Failed to delete collection **{col_name}**.").send()
    await _show_admin_panel()


@cl.action_callback("admin_remove_user_from_group")
async def on_admin_remove_user_from_group(action: cl.Action) -> None:
    payload = action.payload or {}
    username = payload.get("username") or action.value.split("|", 1)[0]
    group_name = payload.get("group_name") or action.value.split("|", 1)[1]
    if not settings.user_auth.enabled:
        return
    from private_gpt.server.auth.user_store import UserStore as _UserStore
    try:
        global_injector.get(_UserStore).remove_user_from_group(username, group_name)
        await cl.Message(content=f"Removed **{username}** from group **{group_name}**.").send()
    except Exception:
        logger.error(
            "Failed to remove user=%s from group=%s", username, group_name, exc_info=True
        )
        await cl.Message(content="Operation failed.").send()
    await _show_admin_panel()


@cl.action_callback("admin_remove_collection_from_group")
async def on_admin_remove_collection_from_group(action: cl.Action) -> None:
    payload = action.payload or {}
    group_name = payload.get("group_name") or action.value.split("|", 1)[0]
    col_name = payload.get("collection_name") or action.value.split("|", 1)[1]
    if not settings.user_auth.enabled:
        return
    from private_gpt.server.auth.user_store import UserStore as _UserStore
    try:
        global_injector.get(_UserStore).remove_collection_from_group(group_name, col_name)
        await cl.Message(
            content=f"Removed collection **{col_name}** from group **{group_name}**."
        ).send()
    except Exception:
        logger.error(
            "Failed to remove col=%s from group=%s", col_name, group_name, exc_info=True
        )
        await cl.Message(content="Operation failed.").send()
    await _show_admin_panel()


# ── Admin: wizard starter action callbacks ────────────────────────────────────

@cl.action_callback("admin_create_user_start")
async def on_admin_create_user_start(action: cl.Action) -> None:
    cl.user_session.set("admin_flow", {"type": "create_user", "step": "username"})
    res = await cl.AskUserMessage(
        content="Enter the **username** for the new user:", timeout=120
    ).send()
    if res is None:
        cl.user_session.set("admin_flow", None)
        return
    await _admin_flow_continue(res["output"].strip())


@cl.action_callback("admin_create_group_start")
async def on_admin_create_group_start(action: cl.Action) -> None:
    cl.user_session.set("admin_flow", {"type": "create_group", "step": "group_name"})
    res = await cl.AskUserMessage(content="Enter the **group name**:", timeout=120).send()
    if res is None:
        cl.user_session.set("admin_flow", None)
        return
    await _admin_flow_continue(res["output"].strip())


@cl.action_callback("admin_create_collection_start")
async def on_admin_create_collection_start(action: cl.Action) -> None:
    cl.user_session.set(
        "admin_flow", {"type": "create_collection", "step": "collection_name"}
    )
    res = await cl.AskUserMessage(
        content="Enter the **collection name** (slug, no spaces):", timeout=120
    ).send()
    if res is None:
        cl.user_session.set("admin_flow", None)
        return
    await _admin_flow_continue(res["output"].strip())


@cl.action_callback("admin_add_user_to_group_start")
async def on_admin_add_user_to_group_start(action: cl.Action) -> None:
    payload = action.payload or {}
    username = payload.get("username") or action.value
    cl.user_session.set(
        "admin_flow", {"type": "add_user_to_group", "step": "group_name", "username": username}
    )
    res = await cl.AskUserMessage(
        content=f"Enter the **group name** to add **{username}** to:", timeout=120
    ).send()
    if res is None:
        cl.user_session.set("admin_flow", None)
        return
    await _admin_flow_continue(res["output"].strip())


@cl.action_callback("admin_add_collection_to_group_start")
async def on_admin_add_collection_to_group_start(action: cl.Action) -> None:
    payload = action.payload or {}
    group_name = payload.get("group_name") or action.value
    cl.user_session.set(
        "admin_flow",
        {"type": "add_collection_to_group", "step": "collection_name", "group_name": group_name},
    )
    res = await cl.AskUserMessage(
        content=f"Enter the **collection name** to assign to group **{group_name}**:",
        timeout=120,
    ).send()
    if res is None:
        cl.user_session.set("admin_flow", None)
        return
    await _admin_flow_continue(res["output"].strip())


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
            content=(
                f"Ingested {len(ingested_names)} file(s): {', '.join(ingested_names)}\n\n"
                "Switch to the **Files** tab in the sidebar to manage your documents."
            )
        ).send()

    return ingested_names


# ── Message handler ────────────────────────────────────────────────────────────

@cl.on_message
async def on_message(message: cl.Message) -> None:
    chat_profile: str = cl.user_session.get("chat_profile") or "Chat"

    # Handle file attachments first (available in all profiles)
    if message.elements:
        await _handle_file_uploads(message.elements)
        if not message.content.strip():
            return

    # ── Files profile ──────────────────────────────────────────────────────────
    if chat_profile == "Files":
        content = message.content.strip().lower()
        # Any message in Files profile refreshes the file list
        await _show_file_manager()
        return

    # ── Admin profile ──────────────────────────────────────────────────────────
    if chat_profile == "Admin":
        is_admin = cl.user_session.get("is_admin", False)
        if not is_admin and settings.user_auth.enabled:
            await cl.Message(
                content="⛔ **Access denied.** Administrator privileges are required."
            ).send()
        else:
            await _show_admin_panel()
        return

    # ── Chat profile (default) ─────────────────────────────────────────────────
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
                "Switch to the **Files** tab in the sidebar, choose a document, "
                "then come back here to summarize it."
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
