import pytest

from tests.ui.workbench_script import requires_node, run

pytestmark = requires_node


SIDEBAR_SETUP = """
const app = el(".app");
const button = el("#sidebarToggleButton");
document.append(app, button);
const view = () => ({
  collapsed: app.classList.contains("sidebar-collapsed"),
  label: button.getAttribute("aria-label"),
  title: button.title,
  expanded: button.getAttribute("aria-expanded"),
});
"""


def test_sidebar_collapses_to_a_rail_and_back() -> None:
    result = run(
        ["$", "applySidebarCollapsed"],
        SIDEBAR_SETUP
        + """
        globalThis.state = { sidebarCollapsed: true };
        applySidebarCollapsed();
        const collapsed = view();
        state.sidebarCollapsed = false;
        applySidebarCollapsed();
        return { collapsed, expanded: view() };
        """,
    )

    assert result == {
        "collapsed": {
            "collapsed": True,
            "label": "Expand sidebar",
            "title": "Expand sidebar",
            "expanded": "false",
        },
        "expanded": {
            "collapsed": False,
            "label": "Collapse sidebar",
            "title": "Collapse sidebar",
            "expanded": "true",
        },
    }


MODEL_LIMITS = [
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "selectedModelInfo",
    "chatMaxTokens",
    "contextWindowSize",
]


@pytest.mark.parametrize(
    ("chat_setting", "advertised", "expected"),
    [
        (256, 8192, 256),
        (None, 8192, 8192),
        (None, None, 4096),
        (0, 8192, 8192),
        ("not a number", None, 4096),
    ],
)
def test_max_output_tokens_prefers_the_chat_then_the_model(
    chat_setting: object, advertised: object, expected: int
) -> None:
    result = run(
        MODEL_LIMITS,
        f"""
        globalThis.state = {{
          selectedModel: "m",
          models: [{{ id: "m", max_tokens: {_js(advertised)} }}],
        }};
        return chatMaxTokens({{ settings: {{ maxTokens: {_js(chat_setting)} }} }});
        """,
    )

    assert result == expected


@pytest.mark.parametrize(("advertised", "expected"), [(64000, 64000), (None, None)])
def test_context_window_comes_from_the_selected_model(
    advertised: object, expected: object
) -> None:
    result = run(
        MODEL_LIMITS,
        f"""
        globalThis.state = {{
          selectedModel: "m",
          models: [{{ id: "other", max_input_tokens: 1 }}, {{ id: "m", max_input_tokens: {_js(advertised)} }}],
        }};
        return contextWindowSize();
        """,
    )

    assert result == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("70", "70"), ("1234", "1.2k"), ("12400", "12k"), ("64000", "64k"), ("NaN", "—")],
)
def test_token_counts_are_abbreviated(value: str, expected: str) -> None:
    assert run(["formatTokenCount"], f"return formatTokenCount({value});") == expected


METER = [
    "$",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "selectedModelInfo",
    "contextWindowSize",
    "formatTokenCount",
    "renderContextMeter",
]


def _meter(used: object, limit: object, messages: int = 1) -> dict[str, object]:
    return run(
        METER,
        f"""
        const meter = el("#contextMeter");
        const label = el("#contextMeterLabel");
        const fill = el("#contextMeterFill", {{ style: {{}} }});
        document.append(meter, label, fill);
        const chat = {{ contextTokens: {"undefined" if used is None else used}, messages: Array({messages}).fill({{ apiMessage: {{}} }}) }};
        globalThis.state = {{ selectedModel: "m", models: [{{ id: "m", max_input_tokens: {_js(limit)} }}] }};
        globalThis.activeChat = () => chat;
        globalThis.sanitizeMessages = messages => messages.filter(m => m.apiMessage);
        renderContextMeter();
        return {{
          hidden: meter.hidden,
          disabled: meter.disabled,
          warn: meter.classList.contains("warn"),
          danger: meter.classList.contains("danger"),
          label: label.textContent,
          fill: fill.style.width ?? null,
        }};
        """,
    )


@pytest.mark.parametrize(
    ("used", "warn", "danger", "label", "fill"),
    [
        (32000, False, False, "32k / 64k", "50%"),
        (48000, True, False, "48k / 64k", "75%"),
        (57600, False, True, "58k / 64k", "90%"),
        (90000, False, True, "90k / 64k", "100%"),
    ],
)
def test_context_meter_warns_as_the_window_fills(
    used: int, warn: bool, danger: bool, label: str, fill: str
) -> None:
    assert _meter(used, 64000) == {
        "hidden": False,
        "disabled": False,
        "warn": warn,
        "danger": danger,
        "label": label,
        "fill": fill,
    }


def test_context_meter_before_anything_is_measured() -> None:
    result = _meter(None, 64000, messages=0)

    # count_tokens rejects an empty conversation, so the recount stays disabled.
    assert result["hidden"] is False
    assert result["disabled"] is True
    assert result["label"] == "— / 64k"


def test_context_meter_is_hidden_when_the_model_advertises_no_window() -> None:
    assert _meter(1000, None)["hidden"] is True


SEND_SETUP = """
const button = el("#sendButton");
document.append(button);
const chat = { id: "c1" };
globalThis.activeChat = () => chat;
const view = () => ({ label: button.getAttribute("aria-label"), stopping: button.classList.contains("stopping"), disabled: button.disabled });
"""


@pytest.mark.parametrize(
    ("sending", "model", "expected"),
    [
        (False, '"m"', {"label": "Send", "stopping": False, "disabled": False}),
        (False, "null", {"label": "Send", "stopping": False, "disabled": True}),
        # Stop must stay usable even if the model list changed mid-request.
        (
            True,
            "null",
            {"label": "Stop generating", "stopping": True, "disabled": False},
        ),
    ],
)
def test_send_button_becomes_stop_while_streaming(
    sending: bool, model: str, expected: dict[str, object]
) -> None:
    result = run(
        ["$", "renderSendButton"],
        SEND_SETUP
        + f"""
        globalThis.runtime = {{ sendingChatIds: new Set({'["c1"]' if sending else "[]"}) }};
        globalThis.state = {{ selectedModel: {model} }};
        renderSendButton();
        return view();
        """,
    )

    assert result == expected


def test_stop_aborts_only_that_chats_request() -> None:
    result = run(
        ["stopGenerating"],
        """
        const first = new AbortController();
        const second = new AbortController();
        globalThis.runtime = { abortControllers: new Map([["c1", first], ["c2", second]]) };
        const stopped = stopGenerating("c1");
        const nothingToStop = stopGenerating("c3");
        return { stopped, nothingToStop, first: first.signal.aborted, second: second.signal.aborted };
        """,
    )

    assert result == {
        "stopped": True,
        "nothingToStop": False,
        "first": True,
        "second": False,
    }


EDIT_SETUP = """
globalThis.toasts = [];
globalThis.requests = [];
globalThis.toast = (text, kind) => toasts.push({ text, kind });
globalThis.touchChat = () => {};
globalThis.saveState = () => {};
globalThis.render = () => {};
globalThis.renderMessages = () => {};
globalThis.apiErrorMessage = error => error.message;
globalThis.requestAssistant = async chat => { requests.push(chat.messages.map(m => m.id)); };
const chat = { id: "c1", messages: [
  { id: "q1", role: "user", text: "first question" },
  { id: "a1", role: "assistant", text: "first answer" },
  { id: "q2", role: "user", text: "second questoin" },
  { id: "a2", role: "assistant", text: "answer to the typo" },
  { id: "q3", role: "user", text: "follow-up" },
] };
globalThis.state = { chats: [chat] };
globalThis.runtime = { sendingChatIds: new Set(), editingMessageId: "q2", expandedMessages: new Set(["q2"]) };
"""


def test_saving_an_edit_drops_later_turns_and_resends() -> None:
    result = run(
        ["saveMessageEdit"],
        EDIT_SETUP
        + """
        document.querySelector = selector => selector === '[data-message-id="q2"] .message-edit-input'
          ? { value: "  second question  " }
          : null;
        saveMessageEdit("c1", "q2");
        await new Promise(resolve => setTimeout(resolve, 0));
        const edited = chat.messages[2];
        return {
          ids: chat.messages.map(m => m.id),
          text: edited.text,
          apiMessage: edited.apiMessage,
          requests,
          editing: runtime.editingMessageId,
          sending: [...runtime.sendingChatIds],
        };
        """,
    )

    assert result == {
        "ids": ["q1", "a1", "q2"],
        "text": "second question",
        "apiMessage": {"role": "user", "content": "second question"},
        "requests": [["q1", "a1", "q2"]],
        "editing": None,
        "sending": [],
    }


@pytest.mark.parametrize(
    ("value", "sending", "toasts"),
    [
        ("   ", False, [{"text": "Message cannot be empty.", "kind": "error"}]),
        ("second question", True, []),
    ],
)
def test_an_edit_is_not_saved_when_empty_or_while_streaming(
    value: str, sending: bool, toasts: list[dict[str, str]]
) -> None:
    result = run(
        ["saveMessageEdit"],
        EDIT_SETUP
        + f"""
        document.querySelector = () => ({{ value: {_js(value)} }});
        if ({_js(sending)}) runtime.sendingChatIds.add("c1");
        saveMessageEdit("c1", "q2");
        return {{ ids: chat.messages.map(m => m.id), text: chat.messages[2].text, requests, toasts }};
        """,
    )

    assert result == {
        "ids": ["q1", "a1", "q2", "a2", "q3"],
        "text": "second questoin",
        "requests": [],
        "toasts": toasts,
    }


def test_stream_keeps_usage_and_forwards_the_abort_signal() -> None:
    result = run(
        ["safeJson", "apiStreamFetch"],
        """
        globalThis.DEFAULT_BASE_URL = "http://pgpt.test";
        globalThis.state = {};
        globalThis.createBasicAuthHeader = () => null;
        globalThis.redactHeaders = headers => headers;
        globalThis.addDebugEvent = () => null;
        globalThis.updateDebugEvent = () => {};
        const events = [
          { type: "message_start", message: { id: "msg_1", role: "assistant", usage: { input_tokens: 812, output_tokens: 1 } } },
          { type: "content_block_start", index: 0, content_block: { type: "text", text: "" } },
          { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "Hello" } },
          { type: "message_delta", delta: { stop_reason: "end_turn" }, usage: { output_tokens: 57 } },
        ];
        const sse = events.map(event => `data: ${JSON.stringify(event)}\\n\\n`).join("");
        let seenSignal = null;
        globalThis.fetch = async (url, options) => {
          seenSignal = options.signal;
          return new Response(sse, { status: 200 });
        };
        const controller = new AbortController();
        const message = await apiStreamFetch("/v1/messages", { body: {}, signal: controller.signal });
        return {
          usage: message.usage,
          stopReason: message.stop_reason,
          text: message.content[0].text,
          signalForwarded: seenSignal === controller.signal,
        };
        """,
    )

    assert result == {
        "usage": {"input_tokens": 812, "output_tokens": 57},
        "stopReason": "end_turn",
        "text": "Hello",
        "signalForwarded": True,
    }


def _js(value: object) -> str:
    """Render a Python test parameter as a JavaScript literal."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return str(value)
