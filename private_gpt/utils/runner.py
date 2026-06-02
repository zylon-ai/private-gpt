def get_version() -> str:
    with open("version.txt", "r+") as version_file:
        try:
            release_version = version_file.read()
            version_file.close()
            return release_version
        except OSError as e:
            raise ImportError("Version file not found") from e
