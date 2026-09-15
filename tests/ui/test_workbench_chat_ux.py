import re
import shutil
import subprocess

import pytest

from tests.ui.workbench_script import inline_script, requires_node, run

pytestmark = requires_node

SCROLL = [
    "$",
    "STICK_TO_BOTTOM_THRESHOLD",
    "messagesAtBottom",
    "syncStickToBottom",
    "renderJumpToLatest",
    "scrollMessagesToBottom",
]

SCROLL_SETUP = """
globalThis.runtime = { stickToBottom: true };
globalThis.updateMessagesFade = () => {};
const messages = el("#messages", { scrollHeight: 2000, clientHeight: 500, scrollTop: 1500 });
const jump = el("#jumpToLatestButton");
document.append(messages, jump);
"""


def test_inline_script_parses() -> None:
    node = shutil.which("node")
    assert node
    completed = subprocess.run(
        [node, "-e", "new Function(require('fs').readFileSync(0, 'utf8'))"],
        input=inline_script(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_new_content_does_not_pull_a_reader_who_scrolled_up() -> None:
    result = run(
        SCROLL,
        SCROLL_SETUP
        + """
        messages.scrollTop = 0;
        syncStickToBottom();
        const jumpHiddenAfterScroll = jump.hidden;
        messages.scrollHeight = 2300;  // a streamed token arrives
        scrollMessagesToBottom();
        return { stick: runtime.stickToBottom, scrollTop: messages.scrollTop, jumpHiddenAfterScroll };
        """,
    )

    assert result == {"stick": False, "scrollTop": 0, "jumpHiddenAfterScroll": False}


def test_new_content_follows_a_reader_at_the_bottom() -> None:
    result = run(
        SCROLL,
        SCROLL_SETUP
        + """
        syncStickToBottom();
        messages.scrollHeight = 2300;
        scrollMessagesToBottom();
        return { stick: runtime.stickToBottom, scrollTop: messages.scrollTop, jumpHidden: jump.hidden };
        """,
    )

    assert result == {"stick": True, "scrollTop": 2300, "jumpHidden": True}


@pytest.mark.parametrize(("distance", "sticks"), [(0, True), (32, True), (33, False)])
def test_scrolling_back_near_the_bottom_resumes_following(
    distance: int, sticks: bool
) -> None:
    result = run(
        SCROLL,
        SCROLL_SETUP
        + f"""
        messages.scrollTop = 0;
        syncStickToBottom();
        messages.scrollTop = 1500 - {distance};
        syncStickToBottom();
        return runtime.stickToBottom;
        """,
    )

    assert result is sticks


def test_forced_scroll_returns_to_the_bottom_and_re_sticks() -> None:
    result = run(
        SCROLL,
        SCROLL_SETUP
        + """
        messages.scrollTop = 0;
        syncStickToBottom();
        scrollMessagesToBottom({ force: true });
        return { stick: runtime.stickToBottom, scrollTop: messages.scrollTop, jumpHidden: jump.hidden };
        """,
    )

    assert result == {"stick": True, "scrollTop": 2000, "jumpHidden": True}


def test_jump_control_stays_hidden_without_overflow() -> None:
    result = run(
        SCROLL,
        SCROLL_SETUP
        + """
        messages.scrollHeight = 520;
        runtime.stickToBottom = false;
        renderJumpToLatest();
        return jump.hidden;
        """,
    )

    assert result is True


COLLAPSE = [
    "$",
    "$$",
    "STICK_TO_BOTTOM_THRESHOLD",
    "renderJumpToLatest",
    "COLLAPSE_THRESHOLD_PX",
    "setCollapseButtonState",
    "applyMessageCollapse",
    "toggleMessageCollapse",
]

COLLAPSE_SETUP = """
globalThis.runtime = { stickToBottom: true, expandedMessages: new Set() };
const message = (id, role, textHeight) => {
  const node = el(`.message.${role}`);
  node.dataset.messageId = id;
  const button = el(".msg-action-collapse").append(el(".collapse-label"));
  node.append(el(".message-bubble").append(el(".message-text", { scrollHeight: textHeight })), button);
  return node;
};
const messages = el("#messages", { scrollHeight: 3000, clientHeight: 800, scrollTop: 2200 });
const nodes = {
  tall: message("tall", "user", 900),
  edge: message("edge", "user", 340),
  short: message("short", "user", 120),
  answer: message("answer", "assistant", 900),
};
messages.append(...Object.values(nodes));
document.append(messages, el("#jumpToLatestButton"));
const view = node => {
  const bubble = $(".message-bubble", node);
  const button = $(".msg-action-collapse", node);
  return {
    collapsible: bubble.classList.contains("collapsible"),
    collapsed: bubble.classList.contains("collapsed"),
    title: button.title ?? null,
    ariaLabel: button.getAttribute("aria-label"),
    ariaExpanded: button.getAttribute("aria-expanded"),
    label: $(".collapse-label", button).textContent,
  };
};
"""


def test_long_user_messages_arrive_collapsed() -> None:
    result = run(
        COLLAPSE,
        COLLAPSE_SETUP
        + """
        applyMessageCollapse();
        return Object.fromEntries(Object.entries(nodes).map(([key, node]) => [key, view(node)]));
        """,
    )

    assert result["tall"] == {
        "collapsible": True,
        "collapsed": True,
        "title": "Expand text",
        "ariaLabel": "Expand text",
        "ariaExpanded": "false",
        "label": "Expand",
    }
    assert result["edge"]["collapsible"] is False
    assert result["short"]["collapsible"] is False
    assert result["short"]["collapsed"] is False
    # Answers are what the reader asked for, so they are never folded away.
    assert result["answer"]["collapsible"] is False
    assert result["answer"]["collapsed"] is False


def test_expander_toggles_and_survives_a_re_render() -> None:
    result = run(
        COLLAPSE,
        COLLAPSE_SETUP
        + """
        applyMessageCollapse();
        toggleMessageCollapse("tall");
        const expanded = view(nodes.tall);
        applyMessageCollapse();  // renderMessages() runs this again on every update
        const reRendered = view(nodes.tall);
        toggleMessageCollapse("tall");
        return { expanded, reRendered, collapsedAgain: view(nodes.tall) };
        """,
    )

    assert result["expanded"] == {
        "collapsible": True,
        "collapsed": False,
        "title": "Collapse text",
        "ariaLabel": "Collapse text",
        "ariaExpanded": "true",
        "label": "Collapse",
    }
    assert result["reRendered"] == result["expanded"]
    assert result["collapsedAgain"]["collapsed"] is True


COPY_SETUP = """
globalThis.copied = [];
globalThis.toasts = [];
globalThis.copyText = (text, label) => copied.push({ text, label });
globalThis.toast = (text, kind) => toasts.push({ text, kind });
globalThis.state = { chats: [{ id: "c1", messages: [
  { id: "markdown", role: "assistant", text: "**Bold** and `code`\\n\\n- item" },
  { id: "blocks", role: "assistant", text: "", contentBlocks: [
    { type: "text", text: "first " }, { type: "tool_use", name: "search" }, { type: "text", text: "second" }
  ] },
  { id: "empty", role: "assistant", text: "" },
] }] };
"""


def test_copy_message_copies_the_markdown_not_the_html() -> None:
    result = run(
        ["copyMessage"],
        COPY_SETUP
        + """
        copyMessage("c1", "markdown");
        copyMessage("c1", "blocks");
        copyMessage("c1", "missing");
        return { copied, toasts };
        """,
    )

    assert result == {
        "copied": [
            {"text": "**Bold** and `code`\n\n- item", "label": "Message copied"},
            {"text": "first second", "label": "Message copied"},
        ],
        "toasts": [],
    }


def test_copying_an_empty_message_says_so() -> None:
    result = run(
        ["copyMessage"],
        COPY_SETUP
        + """
        copyMessage("c1", "empty");
        return { copied, toasts };
        """,
    )

    assert result == {
        "copied": [],
        "toasts": [{"text": "Nothing to copy.", "kind": "error"}],
    }


def test_code_blocks_render_with_a_copy_control() -> None:
    result = run(
        ["renderMarkdown", "escapeHtml"],
        """
        return renderMarkdown("```python\\nprint('<hi>')\\n```");
        """,
    )

    assert (
        '<button class="code-copy-button" type="button" data-copy-code '
        'title="Copy code" aria-label="Copy code">Copy</button>'
    ) in result
    assert "<pre><code>print(&#039;&lt;hi&gt;&#039;)\n</code></pre>" in result


def test_chat_width_presets_default_to_wide() -> None:
    result = run(
        ["CHAT_WIDTHS", "DEFAULT_APPEARANCE"],
        "return { widths: CHAT_WIDTHS, fallback: DEFAULT_APPEARANCE.chatWidth };",
    )

    assert result == {
        "widths": {"comfortable": "860px", "wide": "1100px", "full": "100%"},
        "fallback": "wide",
    }


@pytest.mark.parametrize(
    ("requested", "saved", "expected"),
    [("full", "wide", "full"), ("enormous", "comfortable", "comfortable")],
)
def test_unknown_chat_width_falls_back(
    requested: str, saved: str, expected: str
) -> None:
    result = run(
        ["CHAT_WIDTHS", "DEFAULT_APPEARANCE", "normalizeAppearanceSuggestion"],
        f"""
        globalThis.state = {{ uiAppearance: {{ chatWidth: "{saved}" }} }};
        return normalizeAppearanceSuggestion({{ chatWidth: "{requested}" }}).chatWidth;
        """,
    )

    assert result == expected


MODELS_SETUP = """
globalThis.toasts = [];
globalThis.saveState = () => {};
globalThis.render = () => {};
globalThis.toast = (text, kind) => {
  const entry = { text, kind };
  toasts.push(entry);
  return { update: (text, kind) => Object.assign(entry, { text, kind }) };
};
"""


@pytest.mark.parametrize(
    ("selected", "expected"), [("beta", "beta"), ("removed-model", "alpha")]
)
def test_model_list_is_replaced_from_the_server(selected: str, expected: str) -> None:
    result = run(
        ["loadModels"],
        MODELS_SETUP
        + f"""
        globalThis.apiFetch = async () => ({{ data: [{{ id: "alpha" }}, {{ id: "beta" }}] }});
        globalThis.state = {{ models: [{{ id: "removed-model" }}], selectedModel: "{selected}" }};
        await loadModels({{ showToast: false }});
        return {{ models: state.models.map(m => m.id), selected: state.selectedModel, toasts }};
        """,
    )

    assert result == {"models": ["alpha", "beta"], "selected": expected, "toasts": []}


def test_quiet_model_refresh_keeps_the_cache_and_reports_a_failure() -> None:
    result = run(
        ["loadModels"],
        MODELS_SETUP
        + """
        globalThis.apiFetch = async () => { throw new Error("offline"); };
        globalThis.state = { models: [{ id: "cached" }], selectedModel: "cached" };
        await loadModels({ showToast: false });
        return { models: state.models.map(m => m.id), toasts };
        """,
    )

    assert result == {
        "models": ["cached"],
        "toasts": [{"text": "Could not load models: offline", "kind": "error"}],
    }


def test_models_are_refreshed_on_startup() -> None:
    # Top-level statements in the script sit at four spaces; function bodies are deeper.
    assert re.search(
        r"^    loadModels\(\{ showToast: false \}\);$", inline_script(), re.MULTILINE
    )
