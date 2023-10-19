import os
import pathlib
from glob import glob

root_path = pathlib.Path(__file__).parents[1]
# This is to prevent a bug in intellij that uses the wrong working directory
os.chdir(root_path)


def _as_module(fixture_path: str) -> str:
    return fixture_path.replace("/", ".").replace("\\", ".").replace(".py", "")


pytest_plugins = [_as_module(fixture) for fixture in glob("tests/fixtures/[!_]*.py")]
