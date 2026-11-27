import builtins
import io
from collections.abc import Callable
from typing import Any

import pytest


@pytest.fixture
def locale_encoding(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Make text files opened without an ``encoding`` use the given code page.

    That is what Windows does: Python opens them in the ANSI code page (cp1252,
    cp932, ...), while on Linux and macOS the locale is almost always UTF-8, so a
    missing ``encoding="utf-8"`` goes unnoticed there.
    """

    def use(code_page: str) -> None:
        real_open = builtins.open

        def open_in_code_page(
            file: Any,
            mode: str = "r",
            buffering: int = -1,
            encoding: str | None = None,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if encoding is None and "b" not in mode:
                encoding = code_page
            return real_open(file, mode, buffering, encoding, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", open_in_code_page)
        # pathlib's read_text() and open() resolve a missing encoding here.
        monkeypatch.setattr(
            io,
            "text_encoding",
            lambda encoding, stacklevel=2: encoding or code_page,
        )

    return use
