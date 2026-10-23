"""Spans for sandbox backends, applied to every provider and session class.

Sandbox backends talk to their service over HTTP; without these spans a
trace only shows anonymous ``GET``/``POST``/``DELETE`` client calls. Each
backend operation gets one named span (``sandbox.create``, ``sandbox.run``,
``sandbox.read_file``...) with its arguments and outcome, and the HTTP calls
it makes are suppressed: the operation span is what matters for debugging.
"""

from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from private_gpt.observability.tracing import (
    SpanKind,
    is_enabled,
    set_span_attributes,
    set_span_status,
    start_span,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_MAX_TEXT = 4_000

# method -> (span name, input builder, output builder)
_SESSION_METHODS: dict[str, tuple[str, Callable[..., Any], Callable[[Any], Any]]] = {
    "exec": (
        "sandbox.run",
        lambda command, opts=None: {
            "command": _clip(command),
            "timeout": getattr(opts, "timeout", None),
            "cwd": getattr(opts, "cwd", None),
        },
        lambda r: {
            "exit_code": r.exit_code,
            "stdout": _clip(r.stdout),
            "stderr": _clip(r.stderr),
            "execution_time_ms": r.execution_time_ms,
        },
    ),
    "read_file": (
        "sandbox.read_file",
        lambda path: {"path": path},
        lambda r: {"bytes": len(r), "content": _preview(r)},
    ),
    "write_file": (
        "sandbox.write_file",
        lambda path, content: {
            "path": path,
            "bytes": len(content),
            "content": _preview(content),
        },
        lambda _: None,
    ),
    "list_dir": (
        "sandbox.list_dir",
        lambda path: {"path": path},
        lambda r: {"entries": len(r), "names": [_entry(e) for e in r[:200]]},
    ),
    "path_exists": (
        "sandbox.path_exists",
        lambda path: {"path": path},
        lambda r: {"exists": r},
    ),
    "is_dir": ("sandbox.is_dir", lambda path: {"path": path}, lambda r: {"is_dir": r}),
    "chmod": (
        "sandbox.chmod",
        lambda path, mode: {"path": path, "mode": oct(mode)},
        lambda _: None,
    ),
    "get_endpoint": (
        "sandbox.get_endpoint",
        lambda port: {"port": port},
        lambda r: {"found": r is not None},
    ),
    "make_dir": (
        "sandbox.make_dir",
        lambda path, *_, **__: {"path": path},
        lambda _: None,
    ),
    "close": ("sandbox.close", lambda force=False: {"force": force}, lambda _: None),
}

_PROVIDER_METHODS: dict[str, tuple[str, Callable[..., Any], Callable[[Any], Any]]] = {
    "create_session": (
        "sandbox.create",
        lambda user_id=None, timeout=None, bundle_specs=None, **kw: {
            "session_id": kw.get("session_id"),
            "timeout": timeout,
            "mounts": len(bundle_specs or []) + len(kw.get("volumes") or []),
        },
        lambda r: {"session": type(r).__name__},
    ),
    "restore_session": (
        "sandbox.restore",
        lambda session_id, *_, **__: {"session_id": session_id},
        lambda r: {"restored": r is not None},
    ),
    "renew_session": ("sandbox.renew", lambda *_, **__: None, lambda _: None),
    "is_session_alive": (
        "sandbox.is_alive",
        lambda *_, **__: None,
        lambda r: {"alive": r},
    ),
    "kill_session": ("sandbox.kill", lambda *_, **__: None, lambda _: None),
    "delete_session": ("sandbox.delete", lambda *_, **__: None, lambda _: None),
}


def _preview(content: Any) -> Any:
    """File content as text (clipped); binary content only as a marker."""
    if isinstance(content, str):
        return _clip(content)
    if not isinstance(content, (bytes, bytearray)):
        return None
    head = bytes(content[: _MAX_TEXT + 4])
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError as exc:
        # A multi-byte character cut at the end of the slice is still text.
        if exc.start < len(head) - 4:
            return f"<binary, {len(content)} bytes>"
        text = head[: exc.start].decode("utf-8")
    if "\x00" in text:
        return f"<binary, {len(content)} bytes>"
    if len(content) > len(head):
        return text[:_MAX_TEXT] + f"… [{len(content) - _MAX_TEXT} more bytes]"
    return _clip(text)


def _entry(entry: Any) -> Any:
    for attr in ("name", "path"):
        if isinstance(value := getattr(entry, attr, None), str):
            return value
    return entry if isinstance(entry, str) else str(entry)


def _clip(text: Any) -> Any:
    if isinstance(text, str) and len(text) > _MAX_TEXT:
        return text[:_MAX_TEXT] + f"… [{len(text) - _MAX_TEXT} more chars]"
    return text


@contextmanager
def _suppress_http() -> Iterator[None]:
    """Hide the backend's HTTP client spans under the operation span."""
    try:
        from opentelemetry import context as otel_context
        from opentelemetry.instrumentation.utils import (
            _SUPPRESS_HTTP_INSTRUMENTATION_KEY,
        )
    except ImportError:
        yield
        return
    token = otel_context.attach(
        otel_context.set_value(_SUPPRESS_HTTP_INSTRUMENTATION_KEY, True)
    )
    try:
        yield
    finally:
        otel_context.detach(token)


def _wrap(
    method: Callable[..., Any],
    name: str,
    capture: Callable[..., Any],
    output: Callable[[Any], Any],
    provider: str,
) -> Callable[..., Any]:
    if getattr(method, "__pgpt_traced__", False) or not inspect.iscoroutinefunction(
        method
    ):
        return method

    @functools.wraps(method)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not is_enabled():
            return await method(self, *args, **kwargs)
        try:
            captured = capture(*args, **kwargs)
        except Exception:
            captured = None
        with (
            start_span(
                name,
                SpanKind.TOOL,
                input=captured,
                attributes={"pgpt.sandbox.provider": provider},
            ) as span,
            _suppress_http(),
        ):
            result = await method(self, *args, **kwargs)
            try:
                set_span_attributes(span, output=output(result))
                if getattr(result, "success", True) is False:
                    set_span_status(
                        span, error=f"exit code {getattr(result, 'exit_code', '?')}"
                    )
            except Exception:
                pass
            return result

    setattr(wrapper, "__pgpt_traced__", True)  # noqa: B010
    return wrapper


def trace_sandbox_class(cls: type, *, provider: bool) -> None:
    """Wrap the backend methods ``cls`` defines (called by ``__init_subclass__``)."""
    methods = _PROVIDER_METHODS if provider else _SESSION_METHODS
    label = (
        cls.__name__.removesuffix("Provider").removesuffix("Session") or cls.__name__
    )
    for attr, (name, capture, output) in methods.items():
        method = cls.__dict__.get(attr)
        if method is not None:
            setattr(cls, attr, _wrap(method, name, capture, output, label))
