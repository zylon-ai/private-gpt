"""Keep the OpenInference llama-index handler cheap on the event loop.

The handler builds span attributes synchronously when a llama-index span
starts, i.e. on the request's event loop, before the LLM call. Profiling a
chat with a long history and an image showed three hot spots (~20 ms per LLM
call):

* ``process_input`` JSON-dumps every bound argument, including the whole
  message list, into ``input.value``. LLM spans already get the structured
  ``llm.input_messages``, so for LLMs it is skipped.
* ``_detect_llm_provider`` tries several provider imports on every span; it
  is cached per class.
* ``_process_messages`` writes every message of the prompt. Only the system
  messages and the last ``max_input_messages`` are kept; earlier turns are
  on the previous traces of the session, and the count of omitted ones is
  recorded.

Each patch checks the handler still has the expected shape and is skipped
otherwise, so an upgrade of the instrumentation degrades to its defaults.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_patched = False


def patch_llama_index_handler(max_input_messages: int) -> None:
    global _patched
    if _patched:
        return
    try:
        from llama_index.core.base.llms.base import BaseLLM
        from openinference.instrumentation.llama_index import _handler
    except ImportError:
        return

    span_cls = getattr(_handler, "_Span", None)
    if span_cls is None:
        logger.warning("llama-index handler layout changed; perf patches skipped")
        return

    original_input = getattr(span_cls, "process_input", None)
    if original_input is not None:

        def process_input(self: Any, instance: Any, bound_args: Any) -> None:
            if isinstance(instance, BaseLLM):
                return
            original_input(self, instance, bound_args)

        span_cls.process_input = process_input

    detect = getattr(_handler, "_detect_llm_provider", None)
    if detect is not None:
        providers: dict[type, str | None] = {}

        def detect_llm_provider(instance: Any) -> str | None:
            cls = type(instance)
            if cls not in providers:
                providers[cls] = detect(instance)
            return providers[cls]

        setattr(_handler, "_detect_llm_provider", detect_llm_provider)  # noqa: B010

    original_messages = getattr(span_cls, "_process_messages", None)
    input_prefix = getattr(_handler, "LLM_INPUT_MESSAGES", None)
    if original_messages is not None and input_prefix is not None:

        def process_messages(self: Any, prefix: str, *messages: Any) -> None:
            if prefix != input_prefix or len(messages) <= max_input_messages:
                original_messages(self, prefix, *messages)
                return
            system = [m for m in messages if _role(m) == "system"]
            rest = [m for m in messages if _role(m) != "system"]
            kept = rest[-max_input_messages:]
            original_messages(self, prefix, *system, *kept)
            self["pgpt.llm.input_messages.omitted"] = len(rest) - len(kept)

        span_cls._process_messages = process_messages

    _patched = True


def _role(message: Any) -> str:
    role = getattr(message, "role", None)
    return str(getattr(role, "value", role))
