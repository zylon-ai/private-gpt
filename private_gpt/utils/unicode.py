"""Unicode normalization forms of file names and paths.

The same accented name can arrive pre-composed ("é", NFC) or decomposed
("e" + U+0301, NFD): macOS file pickers, zip archives made on macOS and some
document parsers emit NFD, most other sources NFC. Both render identically, but
Linux filesystems, object stores and metadata filters compare names
byte-for-byte, so a name copied from one source misses a file named by another.
"""

import unicodedata


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def unicode_forms(text: str) -> list[str]:
    """Return *text* as given, then in NFC and NFD, without duplicates."""
    return list(dict.fromkeys((text, nfc(text), unicodedata.normalize("NFD", text))))
