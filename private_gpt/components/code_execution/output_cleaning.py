"""Cleaning pipeline applied to code execution output before it reaches the model.

Sandbox output is untrusted noise as often as it is signal: a ``cat`` of an
image, a notebook with embedded PNGs, a colored test runner or a pip progress
bar turns into thousands of tokens that tell the model nothing. Each
interceptor rewrites one kind of noise; the cleaner chains them in order.

Only code execution output goes through here. Other tools (semantic search,
web fetch, MCP...) return content whose exact text matters downstream and must
not be rewritten.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from private_gpt.settings.settings import OutputCleaningSettings


def human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def _base64_decoded_size(encoded: str) -> int:
    payload = re.sub(r"\s", "", encoded)
    return max(len(payload) * 3 // 4 - payload.count("="), 0)


class OutputInterceptor(ABC):
    """Rewrites one kind of noise in tool output.

    Must return the text unchanged when there is nothing to clean.
    """

    name: ClassVar[str]

    @abstractmethod
    def __call__(self, text: str) -> str: ...


class BinaryOutputInterceptor(OutputInterceptor):
    """Replaces output that is mostly binary with a single placeholder.

    Binary bytes arrive as U+FFFD (lenient decode), NUL or other C0 controls;
    when they dominate, nothing in the stream is worth keeping. Otherwise the
    scattered binary runs are collapsed in place. Terminal controls that
    later interceptors understand (ESC, backspace, CR) are not counted.
    """

    name = "binary"
    _CHARS = r"[�\x00-\x07\x0b\x0c\x0e-\x1a\x1c-\x1f\x7f]"
    _BINARY_CHARS = re.compile(_CHARS)
    _BINARY_RUN = re.compile(_CHARS + "{4,}")

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def __call__(self, text: str) -> str:
        if not text:
            return text
        binary = len(self._BINARY_CHARS.findall(text))
        if binary == 0:
            return text
        if binary / len(text) >= self._threshold:
            size = human_size(len(text.encode("utf-8", errors="replace")))
            return f"[binary output omitted: {size}]"
        return self._BINARY_RUN.sub(
            lambda m: f"[binary data omitted: {len(m.group())} bytes]", text
        )


class AnsiEscapeInterceptor(OutputInterceptor):
    """Strips ANSI/VT escape sequences, in 7-bit and 8-bit (C1) forms.

    Covers colors and cursor moves (CSI), window titles and hyperlinks (OSC),
    DCS/SOS/PM/APC strings (sixel, tmux passthrough, kitty graphics),
    charset designations (``ESC ( B``) and single-character escapes
    (``ESC =``, ``ESC 7``). String payloads stop at the first terminator so an
    unterminated sequence cannot swallow the rest of the output.
    """

    name = "ansi"
    _ANSI = re.compile(
        r"(?:\x1b\]|\x9d)[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c)"  # OSC
        r"|(?:\x1b[PX^_]|[\x90\x98\x9e\x9f])[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c)"
        r"|(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]"  # CSI
        r"|\x1b[ -/]+[0-~]"  # nF: charset designation, ESC SP F...
        r"|\x1b[0-~]"  # Fp/Fe/Fs single-character escapes
        r"|\x1b"  # lone ESC left by a truncated sequence
    )

    def __call__(self, text: str) -> str:
        if "\x1b" not in text and not re.search(r"[\x90-\x9f]", text):
            return text
        return self._ANSI.sub("", text)


class TerminalRenderInterceptor(OutputInterceptor):
    r"""Renders carriage returns and backspaces the way a terminal shows them.

    Progress bars redraw a line with ``\r``: only the last frame is kept.
    Man-page style overstrike (``_\bX`` underline, ``X\bX`` bold) keeps the
    final character, like ``col -b``.
    """

    name = "terminal"

    @staticmethod
    def _last_frame(line: str) -> str:
        frames = [frame for frame in line.split("\r") if frame]
        return frames[-1] if frames else ""

    @staticmethod
    def _apply_backspaces(text: str) -> str:
        out: list[str] = []
        for ch in text:
            if ch != "\b":
                out.append(ch)
            elif out and out[-1] != "\n":
                out.pop()
        return "".join(out)

    def __call__(self, text: str) -> str:
        if "\r" in text:
            text = text.replace("\r\n", "\n")
            if "\r" in text:
                text = "\n".join(self._last_frame(line) for line in text.split("\n"))
        if "\b" in text:
            text = self._apply_backspaces(text)
        return text


class ControlCharInterceptor(OutputInterceptor):
    """Drops C0/C1 control characters that survived the earlier passes.

    Tabs and newlines are kept.
    """

    name = "control"
    _CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")

    def __call__(self, text: str) -> str:
        return self._CONTROL.sub("", text)


class InvisibleUnicodeInterceptor(OutputInterceptor):
    """Removes invisible Unicode that can hide or reorder text.

    Bidi overrides and isolates (Trojan Source, CVE-2021-42574), zero-width
    spaces, BOMs and tag characters (used to smuggle hidden prompts) carry no
    meaning for the model. ZWJ/ZWNJ and LRM/RLM are kept: emoji sequences and
    RTL scripts need them. Lone surrogates become U+FFFD.
    """

    name = "unicode"
    _INVISIBLE = re.compile("[​⁠⁡-⁤﻿‪-‮⁦-⁩\U000e0000-\U000e007f]")
    _SURROGATE = re.compile("[\ud800-\udfff]")

    def __call__(self, text: str) -> str:
        if text.isascii():
            return text
        text = self._INVISIBLE.sub("", text)
        return self._SURROGATE.sub("�", text)


class SecretRedactionInterceptor(OutputInterceptor):
    """Redacts credentials printed by commands like ``env`` or ``cat .netrc``.

    Covers the principal tokens injected into the sandbox environment too.
    """

    name = "secrets"
    _PATTERNS: ClassVar[list[tuple[str, re.Pattern[str]]]] = [
        (
            "private key",
            re.compile(
                r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
                re.DOTALL,
            ),
        ),
        ("credentials", re.compile(r"(?<=://)[^:/\s@]+:[^@\s/]+(?=@)")),
        ("github token", re.compile(r"\b(?:gh[pousr]|github_pat)_\w{36,}")),
        ("aws key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
        ("google key", re.compile(r"\bAIza[\w-]{35}\b")),
        ("anthropic key", re.compile(r"\bsk-ant-[\w-]{20,}")),
        ("openai key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}")),
        ("stripe key", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{24,}")),
        ("slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
        ("jwt", re.compile(r"\beyJ[\w-]{8,}\.eyJ[\w-]{8,}\.[\w-]{8,}")),
    ]
    # NAME=value lines (``env``, .env files). The value must look like an opaque
    # literal so code such as ``API_KEY=os.environ["X"]`` is left alone.
    _ASSIGNMENT = re.compile(
        r"(?im)^(\s*(?:export\s+)?[\w.-]*"
        r"(?:TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|CREDENTIALS?)"
        r"[\w.-]*\s*[=:]\s*)[\"']?([^\s\"'$(){}\[\]<>]{8,})[\"']?\s*$"
    )

    def __call__(self, text: str) -> str:
        for kind, pattern in self._PATTERNS:
            text = pattern.sub(f"[REDACTED {kind}]", text)
        return self._ASSIGNMENT.sub(r"\1[REDACTED]", text)


class DataUriInterceptor(OutputInterceptor):
    """Replaces ``data:<mime>;base64,<payload>`` URIs, keeping the media type."""

    name = "data_uri"
    _DATA_URI = re.compile(
        r"data:(?P<mime>[\w.+-]+/[\w.+-]+)?(?:;[\w-]+=[\w.-]+)*;base64,"
        r"(?P<payload>[A-Za-z0-9+/]+={0,2})"
    )

    def __init__(self, min_length: int) -> None:
        self._min_length = min_length

    def _replace(self, match: re.Match[str]) -> str:
        payload = match.group("payload")
        if len(payload) < self._min_length:
            return match.group()
        mime = match.group("mime") or "unknown"
        size = human_size(_base64_decoded_size(payload))
        return f"[base64 {mime} omitted: {size}]"

    def __call__(self, text: str) -> str:
        if "base64," not in text:
            return text
        return self._DATA_URI.sub(self._replace, text)


class Base64BlobInterceptor(OutputInterceptor):
    """Replaces long bare base64 runs, including fixed-width wrapped ones.

    Wrapping covers the ``base64`` CLI, PEM bodies and JSON escaped newlines.
    A run only counts when it mixes upper, lower and digits, so separators
    (``=====``), long hex hashes and identifiers are left alone.
    """

    name = "base64"
    _BLOB = re.compile(
        r"[A-Za-z0-9+/]{16,}(?:(?:\r?\n|\\n)[A-Za-z0-9+/]{16,})*"
        r"(?:(?:\r?\n|\\n)?[A-Za-z0-9+/]{1,15})?={0,2}"
    )

    def __init__(self, min_length: int) -> None:
        self._min_length = min_length

    def _replace(self, match: re.Match[str]) -> str:
        blob = match.group()
        payload = re.sub(r"\r?\n|\\n", "", blob)
        if len(payload) < self._min_length:
            return blob
        if not (
            re.search(r"[A-Z]", payload)
            and re.search(r"[a-z]", payload)
            and re.search(r"[0-9]", payload)
        ):
            return blob
        return f"[base64 data omitted: {human_size(_base64_decoded_size(payload))}]"

    def __call__(self, text: str) -> str:
        if len(text) < self._min_length:
            return text
        return self._BLOB.sub(self._replace, text)


class LongLineInterceptor(OutputInterceptor):
    """Caps each line (minified JS/CSS, single-line JSON dumps)."""

    name = "long_lines"

    def __init__(self, max_length: int) -> None:
        self._max_length = max_length

    def _cap(self, line: str) -> str:
        if len(line) <= self._max_length:
            return line
        omitted = len(line) - self._max_length
        return f"{line[: self._max_length]}... [{omitted} chars truncated]"

    def __call__(self, text: str) -> str:
        if len(text) <= self._max_length:
            return text
        return "\n".join(self._cap(line) for line in text.split("\n"))


class RepeatedLineInterceptor(OutputInterceptor):
    """Collapses runs of identical consecutive lines into one line plus a count.

    Only exact repeats: masking digits to catch "similar" lines would also
    collapse genuine numeric output (CSV rows, ``seq``) the model asked for.
    """

    name = "repeated_lines"

    def __init__(self, min_run: int) -> None:
        self._min_run = min_run

    def __call__(self, text: str) -> str:
        lines = text.split("\n")
        out: list[str] = []
        i = 0
        while i < len(lines):
            j = i
            while j + 1 < len(lines) and lines[j + 1] == lines[i]:
                j += 1
            run = j - i + 1
            if run >= self._min_run and lines[i].strip():
                out.extend(
                    [lines[i], f"... [previous line repeated {run - 1} more times]"]
                )
            else:
                out.extend(lines[i : j + 1])
            i = j + 1
        return "\n".join(out)


class WhitespaceInterceptor(OutputInterceptor):
    """Strips trailing whitespace per line and collapses runs of blank lines."""

    name = "whitespace"
    _TRAILING = re.compile(r"[ \t]+$", re.MULTILINE)
    _BLANK_RUN = re.compile(r"\n{3,}")

    def __call__(self, text: str) -> str:
        text = self._TRAILING.sub("", text)
        return self._BLANK_RUN.sub("\n\n", text)


def _registry(
    settings: OutputCleaningSettings,
) -> dict[str, Callable[[], OutputInterceptor]]:
    return {
        "binary": lambda: BinaryOutputInterceptor(settings.binary_threshold),
        "ansi": AnsiEscapeInterceptor,
        "terminal": TerminalRenderInterceptor,
        "control": ControlCharInterceptor,
        "unicode": InvisibleUnicodeInterceptor,
        "secrets": SecretRedactionInterceptor,
        "data_uri": lambda: DataUriInterceptor(settings.base64_min_length),
        "base64": lambda: Base64BlobInterceptor(settings.base64_min_length),
        "long_lines": lambda: LongLineInterceptor(settings.max_line_length),
        "repeated_lines": lambda: RepeatedLineInterceptor(settings.min_repeated_lines),
        "whitespace": WhitespaceInterceptor,
    }


class OutputCleaner:
    """Runs a sequence of interceptors over tool output, in order."""

    def __init__(self, interceptors: Sequence[OutputInterceptor] = ()) -> None:
        self._interceptors = list(interceptors)

    @classmethod
    def from_names(
        cls, names: Sequence[str], settings: OutputCleaningSettings
    ) -> OutputCleaner:
        if not settings.enabled:
            return cls()
        available = _registry(settings)
        unknown = set(names) - available.keys()
        if unknown:
            raise ValueError(
                f"Unknown output interceptors: {', '.join(sorted(unknown))}. "
                f"Expected any of: {', '.join(available)}"
            )
        return cls([available[name]() for name in names])

    @classmethod
    def for_bash(cls, settings: OutputCleaningSettings) -> OutputCleaner:
        return cls.from_names(settings.bash_interceptors, settings)

    @classmethod
    def for_view(cls, settings: OutputCleaningSettings) -> OutputCleaner:
        return cls.from_names(settings.view_interceptors, settings)

    def __call__(self, text: str) -> str:
        for interceptor in self._interceptors:
            text = interceptor(text)
        return text


def truncate_head_tail(
    text: str, max_bytes: int, head_ratio: float = 0.2, hint: str = ""
) -> str:
    """Keep the start and the end of *text* within *max_bytes*.

    Command output puts its errors and summaries at the end, so the tail gets
    most of the budget. Cuts snap to line boundaries when one is near (a
    single huge line is cut on a UTF-8 character boundary instead) and the
    marker tells the model how much was dropped and, via *hint*, how to get
    at it.
    """
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    head_bytes = int(max_bytes * head_ratio)
    tail_bytes = max_bytes - head_bytes
    head = encoded[:head_bytes].decode("utf-8", errors="ignore")
    tail = encoded[-tail_bytes:].decode("utf-8", errors="ignore") if tail_bytes else ""
    if (cut := head.rfind("\n")) > len(head) // 2:
        head = head[:cut]
    if (cut := tail.find("\n")) != -1 and cut < len(tail) // 2:
        tail = tail[cut + 1 :]
    omitted = text[len(head) : len(text) - len(tail)]
    omitted_lines = len(omitted.strip("\n").splitlines())
    size = human_size(len(omitted.encode("utf-8")))
    detail = f"{omitted_lines} lines, {size}" if omitted_lines > 0 else size
    reason = f"; output exceeded the {human_size(max_bytes)} limit"
    return f"{head}\n... [{detail} truncated{reason}{hint}] ...\n{tail}"


def truncate_lines(text: str, max_bytes: int) -> tuple[str, int]:
    """Keep the leading whole lines of *text* that fit in *max_bytes*.

    Returns the kept text and the number of lines kept. At least one line is
    always kept (cut on a UTF-8 boundary if it alone exceeds the budget).
    """
    if len(text.encode("utf-8")) <= max_bytes:
        return text, len(text.splitlines())
    kept: list[str] = []
    used = 0
    for line in text.splitlines():
        size = len(line.encode("utf-8")) + 1
        if used + size > max_bytes:
            break
        kept.append(line)
        used += size
    if not kept:
        first = text.splitlines()[0].encode("utf-8")[:max_bytes]
        return first.decode("utf-8", errors="ignore"), 1
    return "\n".join(kept), len(kept)
