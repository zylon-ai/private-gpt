import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_GLOBAL_KEY = "_global"


class ContextManager:
    """Manages embedding context descriptor files stored on disk.

    Context descriptors are plain-text files that explain abbreviations,
    technical terms, and document structure. They are injected as metadata
    into every chunk during ingestion so that the embedding model can form
    semantic connections between technical codes and their plain-language
    equivalents (e.g. "L1" ↔ "cooler length").

    Files are stored under ``<local_data>/context/``, one file per
    collection name. Documents ingested without a collection use the
    global context (stored as ``_global.txt``).
    """

    def __init__(self, base_path: Path) -> None:
        self._base = base_path / "context"
        self._base.mkdir(parents=True, exist_ok=True)

    def _path(self, collection_name: str | None) -> Path:
        key = collection_name if collection_name else _GLOBAL_KEY
        return self._base / f"{key}.txt"

    def save(self, context_text: str, collection_name: str | None = None) -> None:
        """Persist a context descriptor for the given collection (or global)."""
        path = self._path(collection_name)
        path.write_text(context_text, encoding="utf-8")
        logger.info(
            "Saved embedding context for collection=%s (%d chars)",
            collection_name,
            len(context_text),
        )

    def load(self, collection_name: str | None = None) -> str | None:
        """Load the context for the given collection.

        Falls back to the global context when a collection-specific context
        does not exist and ``collection_name`` is provided.  Returns ``None``
        when no context is available.
        """
        path = self._path(collection_name)
        if path.exists():
            return path.read_text(encoding="utf-8")
        # Fall back to global context if a collection-specific one is missing
        if collection_name is not None:
            global_path = self._path(None)
            if global_path.exists():
                logger.debug(
                    "No collection-specific context for collection=%s, using global",
                    collection_name,
                )
                return global_path.read_text(encoding="utf-8")
        return None

    def load_exact(self, collection_name: str | None = None) -> str | None:
        """Load the context for the given collection without falling back to global.

        Returns ``None`` if no context is stored for exactly this collection.
        """
        path = self._path(collection_name)
        return path.read_text(encoding="utf-8") if path.exists() else None

    def delete(self, collection_name: str | None = None) -> bool:
        """Remove the context for the given collection. Returns True if deleted."""
        path = self._path(collection_name)
        if path.exists():
            path.unlink()
            logger.info("Deleted embedding context for collection=%s", collection_name)
            return True
        return False

    def list_all(self) -> list[dict]:
        """Return all stored contexts as a list of dicts."""
        results = []
        for f in sorted(self._base.glob("*.txt")):
            stem = f.stem
            results.append(
                {
                    "collection_name": None if stem == _GLOBAL_KEY else stem,
                    "context_text": f.read_text(encoding="utf-8"),
                }
            )
        return results
