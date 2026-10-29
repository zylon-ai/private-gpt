from __future__ import annotations

import asyncio
import posixpath
import re
import shlex
from typing import TYPE_CHECKING

from private_gpt.components.code_execution.base import CodeExecutionSession
from private_gpt.components.code_execution.results import (
    BashExecutionResult,
    FileOperationResult,
)
from private_gpt.components.sandbox.base import SandboxExecOptions
from private_gpt.observability.tracing import SpanKind, set_span_status, traced
from private_gpt.utils.unicode import unicode_forms

if TYPE_CHECKING:
    from private_gpt.components.environment.environment import Environment
    from private_gpt.components.sandbox.base import SandboxLink, SandboxSession


def _as_text(raw: bytes) -> str | None:
    """Decode *raw* as UTF-8 text, or return None when it is binary.

    A lenient decode turns an image into pages of replacement characters that
    cost tokens and tell the model nothing. NUL bytes are rejected as well: they
    are valid UTF-8 but never appear in a text file.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return None if "\x00" in text else text


# Ask CLIs for plain output at the source; the output cleaner only has to
# catch what still slips through. Exported as a shell prefix rather than via
# SandboxExecOptions.env (not every backend honours it), so an ``export``
# inside the model's own command still wins.
PLAIN_OUTPUT_ENV: dict[str, str] = {
    "NO_COLOR": "1",
    "FORCE_COLOR": "0",
    "CLICOLOR": "0",
    "CLICOLOR_FORCE": "0",
    "TERM": "dumb",
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "GH_PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "COLORTERM": "",
    "MANPAGER": "cat",
    "SYSTEMD_PAGER": "",
    "PY_COLORS": "0",
    "PIP_PROGRESS_BAR": "off",
    "PIP_NO_COLOR": "1",
    "NPM_CONFIG_COLOR": "false",
    "NPM_CONFIG_PROGRESS": "false",
    "TQDM_DISABLE": "1",
    "CARGO_TERM_COLOR": "never",
    "CARGO_TERM_PROGRESS_WHEN": "never",
    "BUILDKIT_PROGRESS": "plain",
    "GCC_COLORS": "",
    "DEBIAN_FRONTEND": "noninteractive",
}
_PLAIN_OUTPUT_PREFIX = "export " + " ".join(
    f"{k}={shlex.quote(v)}" for k, v in PLAIN_OUTPUT_ENV.items()
)


# Shell words of a command: a quoted string or a run of non-separator characters.
_SHELL_WORD = re.compile(r""""([^"\n]*)"|'([^'\n]*)'|([^\s'"`;|&<>()]+)""")
# ``NAME=value`` / ``--flag=value``: only the value can be a path.
_ASSIGNMENT_PREFIX = re.compile(r"^-{0,2}[A-Za-z_][\w-]*=")
_FILE_SUFFIX = re.compile(r"\.\w{1,8}$")
# Bound on the words probed per command, so a long script does not turn into
# hundreds of filesystem round trips.
_MAX_PATH_PROBES = 16


def _path_candidates(command: str, offset: int = 0) -> list[tuple[int, int, str]]:
    """Return ``(start, end, word)`` for each word that may name an accented file.

    Only words that change under Unicode normalization can be misspelled this
    way, and only path-looking ones (a ``/`` or a file extension) are kept.
    Quoted strings are also searched inside, since code passed inline
    (``python -c 'open("…")'``) nests its own quotes; an outer word always
    comes before the words inside it.
    """
    candidates: list[tuple[int, int, str]] = []
    for match in _SHELL_WORD.finditer(command):
        group = next(g for g in (1, 2, 3) if match.group(g) is not None)
        start, end, word = match.start(group), match.end(group), match.group(group)
        if group == 3 and (prefix := _ASSIGNMENT_PREFIX.match(word)):
            start, word = start + prefix.end(), word[prefix.end() :]
        if len(unicode_forms(word)) == 1:
            continue
        if "/" in word or _FILE_SUFFIX.search(word):
            candidates.append((offset + start, offset + end, word))
        if group != 3:
            candidates.extend(_path_candidates(word, offset + start))
    return candidates


def _file_result(result: FileOperationResult) -> dict[str, object]:
    """Span output for a file operation; failures are marked on the span."""
    if not result.success:
        set_span_status(error=result.error or "File operation failed")
    return result.model_dump(exclude_none=True)


class SandboxCodeExecutionSession(CodeExecutionSession):
    """CodeExecutionSession tool protocol over a managed Environment.

    The environment owns lifetime/idle tracking; its sandbox owns path
    translation and permission enforcement. This class only adapts the
    tool protocol on top.
    """

    def __init__(self, environment: Environment) -> None:
        self._env = environment
        self._id = environment.id

    @property
    def _sandbox(self) -> SandboxSession:
        return self._env.sandbox

    def _resolve_path(self, path: str) -> str:
        if posixpath.isabs(path):
            return path
        return posixpath.join(self._env.workspace, path)

    async def _find(self, path: str) -> str | None:
        """Return the Unicode form of absolute *path* that exists, if any.

        An accented name can be stored pre-composed ("é") or decomposed ("e" +
        U+0301). Both render the same, so a model copying the name from search
        results or a file listing may produce the other form, which the
        sandbox filesystem treats as a different, missing file.
        """
        for form in unicode_forms(path):
            if await self._sandbox.path_exists(form):
                return form
        return None

    async def locate(self, path: str) -> str:
        path = self._resolve_path(path)
        if len(unicode_forms(path)) == 1:
            return path
        return await self._find(path) or path

    async def _locate_paths(self, command: str) -> str:
        """Respell accented file paths in *command* the way they exist on disk.

        A word is only replaced when it names no file as given but another
        Unicode form of it does, so commands that already work are untouched.
        """
        candidates = _path_candidates(command)[:_MAX_PATH_PROBES]
        if not candidates:
            return command

        async def respelled(word: str) -> str:
            try:
                if await self._sandbox.path_exists(self._resolve_path(word)):
                    return word
                for form in unicode_forms(word)[1:]:
                    if await self._sandbox.path_exists(self._resolve_path(form)):
                        return form
            except Exception:
                # Not a path inside the sandbox mounts; leave it as written.
                pass
            return word

        replacements = await asyncio.gather(
            *(respelled(word) for _, _, word in candidates)
        )
        # Keep the outermost respelled word of each nested group.
        chosen: list[tuple[int, int, str]] = []
        for (start, end, word), replacement in zip(
            candidates, replacements, strict=True
        ):
            if replacement == word:
                continue
            if any(
                start < kept_end and kept_start < end
                for kept_start, kept_end, _ in chosen
            ):
                continue
            chosen.append((start, end, replacement))
        for start, end, replacement in sorted(chosen, reverse=True):
            command = command[:start] + replacement + command[end:]
        return command

    @traced(
        "sandbox.exec",
        SpanKind.TOOL,
        capture=lambda _self, command, timeout=None, restart=False: {
            "command": command,
            "timeout": timeout,
            "restart": restart,
        },
        result=lambda r: r.model_dump(exclude={"success"}),
    )
    async def execute_bash(
        self, command: str, timeout: int | None = None, restart: bool = False
    ) -> BashExecutionResult:
        workspace = self._env.workspace
        if restart:
            # No cwd: the default is the workspace itself, which every backend
            # can resolve (cwd="/" is outside the local translator's mounts).
            await self._env.exec(f"rm -rf {workspace}* {workspace}.[!.]*")
            await self._sandbox.make_dir(workspace)
        command = await self._locate_paths(command)
        result = await self._env.exec(
            f"{_PLAIN_OUTPUT_PREFIX}\n{command}",
            SandboxExecOptions(timeout=timeout, cwd=workspace),
        )
        return BashExecutionResult(
            success=result.success,
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.exit_code,
            execution_time_ms=result.execution_time_ms,
        )

    @traced(
        "sandbox.view",
        SpanKind.TOOL,
        capture=lambda _self, path, view_range=None, **_: {
            "path": path,
            "view_range": view_range,
        },
        result=_file_result,
    )
    async def view(
        self,
        path: str,
        view_range: tuple[int, int] | None = None,
        include_line_numbers: bool = True,
    ) -> FileOperationResult:
        self._env.touch()
        try:
            path = await self.locate(path)
            if not await self._sandbox.path_exists(path):
                return FileOperationResult(
                    success=False, error=f"File not found: {path}"
                )
            if await self._sandbox.is_dir(path):
                entries = await self._sandbox.list_dir(path)
                return FileOperationResult(success=True, output="\n".join(entries))
            raw = await self._sandbox.read_file(path)
            text = _as_text(raw)
            if text is None:
                return FileOperationResult(
                    success=False,
                    error=(
                        f"{path} is a binary file and cannot be viewed as text. "
                        "Inspect it with code instead, or use describe_image, "
                        "transcribe_audio or convert_documents when available."
                    ),
                )
            all_lines = text.splitlines()
            total = len(all_lines)
            base_line = 1
            view_lines = all_lines
            if view_range is not None:
                start, end = view_range
                start_idx = max(start, 1) - 1
                end_idx = None if end == -1 else max(end, 0)
                view_lines = all_lines[start_idx:end_idx]
                base_line = start_idx + 1
            output = "\n".join(
                f"{i}: {line}" if include_line_numbers else line
                for i, line in enumerate(view_lines, start=base_line)
            )
            return FileOperationResult(success=True, output=output, total_lines=total)
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    @traced(
        "sandbox.str_replace",
        SpanKind.TOOL,
        capture=lambda _self, path, old_str, new_str: {
            "path": path,
            "old_str": old_str,
            "new_str": new_str,
        },
        result=_file_result,
    )
    async def str_replace(
        self, path: str, old_str: str, new_str: str
    ) -> FileOperationResult:
        if not isinstance(old_str, str):
            return FileOperationResult(
                success=False, error="str_replace requires the old_str parameter."
            )
        if not isinstance(new_str, str):
            return FileOperationResult(
                success=False, error="str_replace requires the new_str parameter."
            )
        self._env.touch()
        try:
            path = await self.locate(path)
            raw = await self._sandbox.read_file(path)
            text = raw.decode("utf-8", errors="replace")
            occurrences = text.count(old_str)
            if occurrences == 0:
                return FileOperationResult(
                    success=False, error="old_str was not found in the file."
                )
            if occurrences > 1:
                return FileOperationResult(
                    success=False,
                    error="old_str appears more than once in the file.",
                )
            start_line = text[: text.index(old_str)].count("\n") + 1
            updated = text.replace(old_str, new_str, 1)
            await self._sandbox.write_file(path, updated.encode("utf-8"))
            return FileOperationResult(
                success=True, output=f"Updated {path}", start_line=start_line
            )
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    @traced(
        "sandbox.create",
        SpanKind.TOOL,
        capture=lambda _self, path, file_text: {
            "path": path,
            "bytes": len(file_text or ""),
        },
        result=_file_result,
    )
    async def create(self, path: str, file_text: str) -> FileOperationResult:
        if not isinstance(file_text, str):
            return FileOperationResult(
                success=False, error="create requires the file_text parameter."
            )
        self._env.touch()
        try:
            # Overwrite an existing file named in another Unicode form rather
            # than leaving two visually identical files side by side.
            path = await self.locate(path)
            is_update = await self._sandbox.path_exists(path)
            await self._sandbox.write_file(path, file_text.encode("utf-8"))
            return FileOperationResult(
                success=True,
                output=f"{'Updated' if is_update else 'Created'} {path}",
                is_update=is_update,
            )
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    @traced(
        "sandbox.insert",
        SpanKind.TOOL,
        capture=lambda _self, path, insert_line, new_str: {
            "path": path,
            "insert_line": insert_line,
            "new_str": new_str,
        },
        result=_file_result,
    )
    async def insert(
        self, path: str, insert_line: int, new_str: str
    ) -> FileOperationResult:
        if not isinstance(new_str, str):
            return FileOperationResult(
                success=False, error="insert requires the new_str parameter."
            )
        self._env.touch()
        try:
            path = await self.locate(path)
            raw = await self._sandbox.read_file(path)
            text = raw.decode("utf-8", errors="replace")
            lines = text.splitlines()
            if insert_line < 0 or insert_line > len(lines):
                return FileOperationResult(
                    success=False,
                    error=f"insert_line {insert_line} is out of range.",
                )
            insertion = new_str.splitlines()
            updated_lines = lines[:insert_line] + insertion + lines[insert_line:]
            updated = "\n".join(updated_lines)
            if text.endswith("\n") or new_str.endswith("\n"):
                updated += "\n"
            await self._sandbox.write_file(path, updated.encode("utf-8"))
            return FileOperationResult(success=True, output=f"Updated {path}")
        except Exception as exc:
            return FileOperationResult(success=False, error=str(exc))

    async def get_endpoint(self, port: int) -> SandboxLink | None:
        return await self._sandbox.get_endpoint(port)

    @traced(
        "sandbox.read_file",
        SpanKind.TOOL,
        capture=lambda _self, path: {"path": path},
        result=lambda raw: {"bytes": len(raw)},
    )
    async def read_file(self, path: str) -> bytes:
        self._env.touch()
        return await self._sandbox.read_file(await self.locate(path))

    @traced(
        "sandbox.write_file",
        SpanKind.TOOL,
        capture=lambda _self, path, content: {"path": path, "bytes": len(content)},
        result=lambda _: None,
    )
    async def write_file(self, path: str, content: bytes) -> None:
        self._env.touch()
        await self._sandbox.write_file(await self.locate(path), content)

    async def path_exists(self, path: str) -> bool:
        path = self._resolve_path(path)
        self._env.touch()
        if len(unicode_forms(path)) == 1:
            return await self._sandbox.path_exists(path)
        return await self._find(path) is not None

    async def close(self) -> None:
        await self._sandbox.close()
