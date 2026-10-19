import base64
import os

import pytest

from private_gpt.components.code_execution.output_cleaning import (
    AnsiEscapeInterceptor,
    Base64BlobInterceptor,
    BinaryOutputInterceptor,
    DataUriInterceptor,
    InvisibleUnicodeInterceptor,
    LongLineInterceptor,
    OutputCleaner,
    RepeatedLineInterceptor,
    SecretRedactionInterceptor,
    TerminalRenderInterceptor,
    truncate_head_tail,
)
from private_gpt.settings.settings import OutputCleaningSettings


def _b64(n: int) -> str:
    return base64.b64encode(os.urandom(n)).decode()


def _cleaner() -> OutputCleaner:
    return OutputCleaner.for_bash(OutputCleaningSettings())


def test_plain_text_is_untouched() -> None:
    text = "total 12\n-rw-r--r-- 1 agent agent 42 file.txt\n\tindented\n"
    assert _cleaner()(text) == text


def test_data_uri_is_replaced_with_mime() -> None:
    text = f'<img src="data:image/png;base64,{_b64(3000)}"> done'
    cleaned = _cleaner()(text)
    assert cleaned.startswith('<img src="[base64 image/png omitted: 2.9 KB]"')
    assert cleaned.endswith(" done")


def test_short_data_uri_is_kept() -> None:
    text = "data:text/plain;base64,aGVsbG8="
    assert DataUriInterceptor(256)(text) == text


def test_wrapped_base64_blob_is_replaced() -> None:
    raw = _b64(4000)
    wrapped = "\n".join(raw[i : i + 76] for i in range(0, len(raw), 76))
    cleaned = _cleaner()(f"before\n{wrapped}\nafter")
    assert cleaned == "before\n[base64 data omitted: 3.9 KB]\nafter"


def test_json_escaped_base64_is_replaced() -> None:
    raw = _b64(1000)
    chunks = "\\n".join(raw[i : i + 76] for i in range(0, len(raw), 76))
    cleaned = _cleaner()(f'{{"image": "{chunks}"}}')
    assert cleaned.startswith('{"image": "[base64 data omitted:')


def test_hex_hashes_and_identifiers_are_kept() -> None:
    sha = "a" * 64 + "0123456789abcdef" * 20
    ident = "SomeVeryLongIdentifierWithoutDigits" * 10
    interceptor = Base64BlobInterceptor(256)
    assert interceptor(sha) == sha
    assert interceptor(ident) == ident


def test_mostly_binary_output_is_replaced_entirely() -> None:
    garbage = os.urandom(2048).decode("utf-8", errors="replace")
    assert _cleaner()(garbage).startswith("[binary output omitted:")


def test_scattered_binary_runs_are_collapsed() -> None:
    text = "header " + "x" * 200 + "\x00\x01\x02\x03\x04" + " footer"
    cleaned = BinaryOutputInterceptor(0.1)(text)
    assert "[binary data omitted: 5 bytes]" in cleaned
    assert cleaned.endswith(" footer")


def test_colored_output_is_not_treated_as_binary() -> None:
    text = "".join(f"\x1b[31mE\x1b[0m\x1b[32mok {i}\x1b[0m\n" for i in range(50))
    assert _cleaner()(text) == "".join(f"Eok {i}\n" for i in range(50))


def test_ansi_osc_and_cursor_sequences_are_stripped() -> None:
    text = "\x1b]0;title\x07\x1b[2K\rProgress 100%\x1b[?25h"
    assert AnsiEscapeInterceptor()(text) == "\rProgress 100%"


@pytest.mark.parametrize(
    "sequence",
    [
        "\x1b[38:2:255:0:0m",  # truecolor with colon sub-params
        "\x1b]8;;https://example.com\x1b\\",  # OSC 8 hyperlink, ST terminator
        "\x1bPq#0;2;0;0;0#0~~@@vv\x1b\\",  # DCS sixel
        "\x1b_Gf=100;AAAA\x1b\\",  # APC kitty graphics
        "\x1b(B",  # charset designation
        "\x1b=",  # keypad mode
        "\x9b31m",  # 8-bit CSI
    ],
)
def test_ansi_sequence_variants_are_stripped(sequence: str) -> None:
    assert AnsiEscapeInterceptor()(f"a{sequence}b") == "ab"


def test_progress_bar_keeps_last_frame() -> None:
    text = "Downloading  10%\rDownloading  50%\rDownloading 100%\r\ndone\r\n"
    assert TerminalRenderInterceptor()(text) == "Downloading 100%\ndone\n"


def test_backspace_overstrike_is_resolved() -> None:
    text = "N\bNA\bAM\bME\bE  _\bl_\bs"
    assert TerminalRenderInterceptor()(text) == "NAME  ls"


def test_pip_style_output_is_fully_cleaned() -> None:
    text = (
        "\x1b[?25l\r\x1b[2K   \x1b[90m━━━━━━━━\x1b[0m 0.0/1.0 MB\r"
        "\x1b[2K   \x1b[32m━━━━━━━━\x1b[0m 1.0/1.0 MB\n\x1b[?25h"
        "Successfully installed foo-1.0   \n"
    )
    assert (
        _cleaner()(text) == "   ━━━━━━━━ 1.0/1.0 MB\nSuccessfully installed foo-1.0\n"
    )


def test_invisible_unicode_is_removed_but_joiners_kept() -> None:
    text = "access\u202e\u2066 = admin\u2069\u200b\ufeff \U000e0041 👩\u200d💻"
    assert InvisibleUnicodeInterceptor()(text) == "access = admin  👩\u200d💻"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ANTHROPIC_AUTH_TOKEN=abcd1234efgh5678", "ANTHROPIC_AUTH_TOKEN=[REDACTED]"),
        ("export DB_PASSWORD='hunter2hunter2'", "export DB_PASSWORD=[REDACTED]"),
        (
            "git clone https://bob:s3cret@host/x",
            "git clone https://[REDACTED credentials]@host/x",
        ),
        ("token ghp_" + "a" * 36, "token [REDACTED github token]"),
        ("AKIAABCDEFGHIJKLMNOP", "[REDACTED aws key]"),
    ],
)
def test_secrets_are_redacted(text: str, expected: str) -> None:
    assert SecretRedactionInterceptor()(text) == expected


def test_secret_like_code_is_not_redacted() -> None:
    text = 'API_KEY = os.environ["API_KEY"]\nSORT_KEY=name\npassword: str'
    assert SecretRedactionInterceptor()(text) == text


def test_long_lines_are_capped() -> None:
    text = "short\n" + "x" * 5000
    assert LongLineInterceptor(2000)(text) == (
        "short\n" + "x" * 2000 + "... [3000 chars truncated]"
    )


def test_repeated_lines_are_collapsed() -> None:
    text = "start\n" + "warning: deprecated\n" * 50 + "end"
    assert RepeatedLineInterceptor(3)(text) == (
        "start\nwarning: deprecated\n... [previous line repeated 49 more times]\nend"
    )


def test_short_repeats_and_numeric_rows_are_kept() -> None:
    text = "a\na\n1,2\n1,3\n1,4"
    assert RepeatedLineInterceptor(3)(text) == text


def test_view_cleaner_keeps_file_structure() -> None:
    cleaner = OutputCleaner.for_view(OutputCleaningSettings())
    text = "def f():   \n\n\n\n    return 1\nx\nx\nx\nx\nPASSWORD=hunter2hunter2"
    assert cleaner(text) == text


def test_head_tail_truncation_keeps_the_end() -> None:
    text = "HEAD" + "m" * 10_000 + "TAIL error: boom"
    out = truncate_head_tail(text, 1000)
    assert out.startswith("HEAD")
    assert out.endswith("TAIL error: boom")
    assert "truncated; output exceeded the 1000 B limit] ..." in out
    assert truncate_head_tail("small", 1000) == "small"


def test_head_tail_truncation_cuts_on_line_boundaries() -> None:
    lines = [f"tests/test_{i:04d}.py::test PASSED" for i in range(3000)]
    lines.append("=== 1 failed, 2999 passed in 1.23s ===")
    out = truncate_head_tail("\n".join(lines), 2000).split("\n")
    marker = next(i for i, line in enumerate(out) if line.startswith("... ["))
    kept = out[:marker] + out[marker + 1 :]
    assert all(line in lines for line in kept)
    assert out[-1] == "=== 1 failed, 2999 passed in 1.23s ==="
    assert out[marker].startswith(f"... [{len(lines) - len(kept)} lines, ")


def test_disabled_cleaner_is_identity() -> None:
    cleaner = OutputCleaner.for_bash(OutputCleaningSettings(enabled=False))
    text = f"data:image/png;base64,{_b64(3000)}"
    assert cleaner(text) == text


def test_interceptors_accept_comma_separated_string() -> None:
    settings = OutputCleaningSettings(bash_interceptors="ansi, base64")  # type: ignore[arg-type]
    assert settings.bash_interceptors == ["ansi", "base64"]


def test_unknown_interceptor_is_rejected() -> None:
    with pytest.raises(ValueError, match="nope"):
        OutputCleaner.for_bash(OutputCleaningSettings(bash_interceptors=["nope"]))
