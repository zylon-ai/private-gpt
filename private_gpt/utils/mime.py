def is_magic_available() -> bool:
    try:
        import magic  # noqa: F401

        return True
    except ImportError:
        return False
