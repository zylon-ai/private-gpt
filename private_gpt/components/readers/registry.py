from injector import inject, singleton

_DEFAULT_EXTENSION_READERS: dict[str, list[str]] = {
    # Binary document formats prefer the existing default readers first.
    # Optional alternative readers like MarkItDown can still be tried when available.
    ".pdf": ["markitdown", "docling", "vision"],
    ".pptx": ["markitdown", "pptx2md"],
    ".docx": ["markitdown", "docling"],
    ".xlsx": ["markitdown", "docling"],
    ".xls": ["markitdown"],
    # Text-like formats stay on the text reader pipeline.
    ".md": ["text"],
    ".html": ["text"],
    ".htm": ["text"],
    ".xhtml": ["text"],
    ".xht": ["text"],
    ".shtml": ["text"],
    ".shtm": ["text"],
    ".stm": ["text"],
    ".txt": ["text"],
    ".csv": ["text"],
    ".tsv": ["text"],
    ".psv": ["text"],
    ".eml": ["text"],
}


def _normalize_reader_name(name: str) -> str:
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Reader name cannot be blank.")
    return normalized


def _normalize_extension(extension: str) -> str:
    normalized = extension.strip().lower()
    if not normalized:
        raise ValueError("Extension cannot be blank.")
    return normalized if normalized.startswith(".") else f".{normalized}"


@singleton
class ReaderRegistry:
    @inject
    def __init__(self) -> None:
        self._registry = {
            extension: reader_names.copy()
            for extension, reader_names in _DEFAULT_EXTENSION_READERS.items()
        }

    def register_extension_reader(self, extension: str, reader_name: str) -> None:
        normalized_extension = _normalize_extension(extension)
        normalized_reader_name = _normalize_reader_name(reader_name)
        current = self._registry.get(normalized_extension, [])
        self._registry[normalized_extension] = [
            normalized_reader_name,
            *[name for name in current if name != normalized_reader_name],
        ]

    def register_extension_readers(
        self,
        extension: str,
        reader_names: list[str],
    ) -> None:
        normalized_extension = _normalize_extension(extension)
        normalized_reader_names = [
            _normalize_reader_name(reader_name) for reader_name in reader_names
        ]
        self._registry[normalized_extension] = list(
            dict.fromkeys(normalized_reader_names)
        )

    def unregister_extension_reader(self, extension: str) -> None:
        self._registry.pop(_normalize_extension(extension), None)

    def get_reader_name(self, extension: str | None) -> str | None:
        reader_names = self.get_reader_names(extension)
        return reader_names[0] if reader_names else None

    def get_reader_names(self, extension: str | None) -> list[str]:
        if not extension:
            return []
        return self._registry.get(_normalize_extension(extension), []).copy()

    def get_all_extensions(self) -> set[str]:
        return set(self._registry.keys())
