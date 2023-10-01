# noinspection PyUnresolvedReferences

from glob import glob


def _as_module(fixture_path: str) -> str:
    return fixture_path.replace("/", ".").replace("\\", ".").replace(".py", "")


pytest_plugins = [_as_module(fixture) for fixture in glob("tests/fixtures/[!_]*.py")]
print(pytest_plugins)
