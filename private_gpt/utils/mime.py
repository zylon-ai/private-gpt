def is_magic_available() -> bool:
    try:
        import magic  # noqa: F401  # ty:ignore[unresolved-import]

        return True
    except ImportError:
        return False
