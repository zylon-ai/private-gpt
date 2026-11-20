"""The published ``private-gpt[core]`` install must boot without optional extras.

CI installs every extra, so an unconditional import of an optional package
(e.g. ``celery`` from the ``queue`` extra) slips through and only breaks the
Homebrew / ``uv tool`` / Docker installs, which ship ``core`` alone. Likewise
CI always has libmagic, but a fresh Windows host never does.
"""

import os
import subprocess
import sys
import textwrap

import pytest

# Top-level packages provided only by extras that ``core`` does not include.
NON_CORE_PACKAGES = ("celery", "flower", "pika")

_BLOCKER = textwrap.dedent(
    """
    import importlib.abc
    import sys

    BLOCKED = set(sys.argv[1].split(","))

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    sys.meta_path.insert(0, _Blocker())

    import private_gpt.main  # noqa: F401
    import private_gpt.cli.main  # noqa: F401
    """
)


@pytest.mark.parametrize(
    "blocked",
    [
        pytest.param(",".join(NON_CORE_PACKAGES), id="core-extras-only"),
        # python-magic raises ImportError when the native libmagic is missing:
        # stock Windows, Linux without libmagic1, macOS without Homebrew's.
        pytest.param("magic", id="no-native-libmagic"),
    ],
)
def test_server_app_imports_without_non_core_extras(blocked: str) -> None:
    env = {**os.environ, "PGPT_PROFILES": os.environ.get("PGPT_PROFILES", "test")}
    result = subprocess.run(
        [sys.executable, "-c", _BLOCKER, blocked],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-4000:]
