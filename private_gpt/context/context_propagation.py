"""Carry request-scoped context across broker boundaries as a global bag.

A single mutable dict ("the bag") is the source of truth for request-scoped
context (headers, cookies, ids, …). Components read from and write to it
directly — there is no per-contextvar plumbing. When a job hops to an ARQ or
Celery worker, the whole bag is serialized into the payload and reinstalled
in the worker, so any component reads the same values as the API process.

The bag is itself a ``ContextVar`` so concurrent requests on the same event
loop stay isolated, but it is a *plain dict*: the API middleware replaces the
whole dict per request, and workers restore the transported dict. Because it
is a single value that is copied wholesale, new context keys require no code
changes to the propagation machinery.

Example::

    from private_gpt.context import current_bag

    def handle(request):
        bag = current_bag()
        bag["headers"] = {...}   # middleware
        ...
        # worker job:
        with reinstall(payload["_context"]):
            headers = current_bag()["headers"]
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

ContextBag = dict[str, Any]

_bag: ContextVar[ContextBag | None] = ContextVar("context_bag", default=None)


def current_bag() -> ContextBag:
    """Return the request-scoped context bag for this coroutine/thread.

    Lazily creates an empty bag when none is installed, so read-only callers
    can always access a dict (and writes from middleware land in it).
    """
    bag = _bag.get()
    if bag is None:
        bag = ContextBag()
        _bag.set(bag)
    return bag


def reset_bag() -> None:
    """Drop the current bag so the next ``current_bag()`` call starts fresh.

    Called at the start/end of each HTTP request so no stale context leaks
    between requests running on the same asyncio task/loop.
    """
    _bag.set(None)


def replace_bag(updates: ContextBag) -> ContextBag:
    """Merge *updates* into a copy of the current bag and install it.

    Copy-on-write: the ContextVar is rebound to a *new* dict, so tasks created
    earlier via ``asyncio.create_task`` (which snapshot the ContextVar by
    reference) keep their own copy and never observe this mutation.
    """
    merged = ContextBag(current_bag())
    merged.update(updates)
    _bag.set(merged)
    return merged


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, set, str)):
        return len(value) == 0
    return False


def snapshot() -> ContextBag:
    """Return a copy of the current bag for a job payload.

    Empty values (``None``, ``{}``, ``[]``, ``""`` …) are dropped so an
    anonymous request contributes an empty payload. The result is
    JSON-compatible as long as the bag's values are, which is the case for
    everything currently written (headers / cookies / ids).
    """
    return {k: v for k, v in current_bag().items() if not _is_empty(v)}


@contextmanager
def reinstall(bag: ContextBag | None = None) -> Iterator[None]:
    """Install a transported *bag* as the current bag for the job duration.

    Restores the previous bag (or none) on exit, so a job is isolated from
    the worker's ambient context and concurrent jobs do not leak into each
    other.
    """
    previous = _bag.get()
    _bag.set(dict(bag) if bag is not None else ContextBag())
    try:
        yield
    finally:
        _bag.set(previous)
