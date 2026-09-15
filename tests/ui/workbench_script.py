"""Run pieces of the Workbench UI's inline script under Node.

`ui/index.html` is a single static file with no build step and no JavaScript test
setup, so these helpers lift named top-level declarations out of its `<script>` and
evaluate them in Node against a small fake DOM. Only what a test names is evaluated;
the rest of the script, including its boot code, never runs.
"""

import json
import pathlib
import re
import shutil
import subprocess
from collections.abc import Iterator
from typing import Any

import pytest

INDEX_HTML = pathlib.Path(__file__).parents[2] / "ui" / "index.html"

requires_node = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is required to run the UI script"
)

# A `/` after one of these (or at the start) begins a regex literal, not a division.
_REGEX_PRECEDERS = set("(,=:[!&|?{};+-*%<>~^")
_REGEX_KEYWORDS = re.compile(r"\b(?:return|typeof|case|in|of)\s*$")

FAKE_DOM = r"""
class FakeClassList {
  constructor() { this.names = new Set(); }
  add(...names) { names.forEach(name => this.names.add(name)); }
  remove(...names) { names.forEach(name => this.names.delete(name)); }
  contains(name) { return this.names.has(name); }
  toggle(name, force) {
    const on = force === undefined ? !this.names.has(name) : Boolean(force);
    if (on) this.names.add(name); else this.names.delete(name);
    return on;
  }
}

class FakeElement {
  constructor(selector = "", props = {}) {
    this.id = (selector.match(/#([\w-]+)/) || [])[1] || "";
    this.classList = new FakeClassList();
    (selector.match(/\.[\w-]+/g) || []).forEach(name => this.classList.add(name.slice(1)));
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.hidden = false;
    this.textContent = "";
    this.parentElement = null;
    Object.assign(this, props);
  }
  append(...children) {
    children.forEach(child => { child.parentElement = this; this.children.push(child); });
    return this;
  }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name] ?? null; }
  // Compound selectors only (`#id`, `.a.b`, `#id.a`), which is all the tests need.
  matches(selector) {
    const id = (selector.match(/#([\w-]+)/) || [])[1];
    const classes = (selector.match(/\.[\w-]+/g) || []).map(name => name.slice(1));
    return (!id || this.id === id) && classes.every(name => this.classList.contains(name));
  }
  querySelectorAll(selector) {
    const found = [];
    const walk = node => node.children.forEach(child => {
      if (child.matches(selector)) found.push(child);
      walk(child);
    });
    walk(this);
    return found;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
}

const el = (selector, props) => new FakeElement(selector, props);
const document = new FakeElement();
"""


def inline_script() -> str:
    html = INDEX_HTML.read_text(encoding="utf-8")
    match = re.search(r"<script>([\s\S]*)</script>", html)
    assert match, "inline <script> not found in ui/index.html"
    return match.group(1)


def _skip_string(source: str, i: int) -> int:
    quote = source[i]
    i += 1
    while source[i] != quote:
        i += 2 if source[i] == "\\" else 1
    return i + 1


def _skip_template(source: str, i: int) -> int:
    i += 1
    while source[i] != "`":
        if source[i] == "\\":
            i += 2
        elif source.startswith("${", i):
            i = _code_end(source, i + 2)
        else:
            i += 1
    return i + 1


def _skip_regex(source: str, i: int) -> int:
    i += 1
    in_class = False
    while in_class or source[i] != "/":
        if source[i] == "\\":
            i += 1
        elif source[i] == "[":
            in_class = True
        elif source[i] == "]":
            in_class = False
        i += 1
    i += 1
    while source[i].isalpha():
        i += 1
    return i


def _code(source: str, i: int) -> Iterator[tuple[int, str]]:
    """Yield `(index, char)` for code outside strings, comments and regexes."""
    previous = ""
    while i < len(source):
        char = source[i]
        if source.startswith("//", i):
            i = source.index("\n", i)
        elif source.startswith("/*", i):
            i = source.index("*/", i) + 2
        elif char in "'\"":
            i = _skip_string(source, i)
            previous = char
        elif char == "`":
            i = _skip_template(source, i)
            previous = char
        elif char == "/" and (
            previous in _REGEX_PRECEDERS
            or not previous
            or _REGEX_KEYWORDS.search(source[max(0, i - 12) : i])
        ):
            i = _skip_regex(source, i)
            previous = "/"
        else:
            yield i, char
            if not char.isspace():
                previous = char
            i += 1


def _code_end(source: str, i: int) -> int:
    """Index just past the bracket that closes the code starting at `i`."""
    depth = 0
    for index, char in _code(source, i):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            if depth == 0:
                return index + 1
            depth -= 1
    raise AssertionError("unbalanced brackets in ui/index.html")


def declaration(name: str, source: str | None = None) -> str:
    """Source of the top-level `function` or `const` named `name`."""
    source = inline_script() if source is None else source
    escaped = re.escape(name)
    match = re.search(
        rf"^[ \t]*((?:async[ \t]+)?function[ \t]+{escaped}(?![\w$])"
        rf"|const[ \t]+{escaped}(?![\w$]))",
        source,
        re.MULTILINE,
    )
    assert match, f"{name} is not declared in ui/index.html"
    is_function = "function" in match.group(1)
    depth = 0
    for index, char in _code(source, match.end(1)):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if is_function and char == "}" and depth == 0:
                return source[match.start(1) : index + 1]
        elif not is_function and char == ";" and depth == 0:
            return source[match.start(1) : index + 1]
    raise AssertionError(f"could not find the end of {name}")


def run(names: list[str], body: str) -> Any:
    """Evaluate the named declarations, then `body`, and return what `body` returns.

    `body` is the inside of an async function, so it may `await` and must `return` a
    JSON-serialisable value. The fake DOM (`el`, `document`) is in scope.
    """
    source = inline_script()
    declarations = "\n\n".join(declaration(name, source) for name in names)
    program = (
        f"{FAKE_DOM}\n{declarations}\n"
        f"(async () => {{\n{body}\n}})()"
        ".then(result => console.log(JSON.stringify(result ?? null)))"
        ".catch(error => { console.error(error); process.exit(1); });\n"
    )
    completed = subprocess.run(
        ["node", "-"],
        input=program,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])
