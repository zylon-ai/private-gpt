import os
import pathlib
import subprocess
from glob import glob

root_path = pathlib.Path(__file__).parents[1]
# This is to prevent a bug in intellij that uses the wrong working directory
os.chdir(root_path)


def _as_module(fixture_path: str) -> str:
    return fixture_path.replace("/", ".").replace("\\", ".").replace(".py", "")


pytest_plugins = [_as_module(fixture) for fixture in glob("tests/fixtures/[!_]*.py")]


def pytest_addoption(parser):
    parser.addoption(
        "--git-diff",
        action="store_true",
        default=False,
        help="Only run tests related to files changed in git",
    )
    parser.addoption(
        "--git-target",
        action="store",
        default=None,
        help="Target branch/commit to compare git diff against (e.g., origin/main)",
    )
    parser.addoption(
        "--block",
        action="append",
        default=[],
        help="Only run tests in the specified block(s) / directory (e.g. server, components)",
    )


def pytest_collection_modifyitems(config, items):
    blocks = config.getoption("--block")
    if blocks:
        selected = []
        deselected = []
        for item in items:
            test_file_path = str(
                getattr(item, "path", getattr(item, "fspath", ""))
            ).replace("\\", "/")
            matched = False
            for block in blocks:
                block_pattern = f"/{block}/"
                if (
                    block_pattern in test_file_path
                    or test_file_path.startswith(f"tests/{block}/")
                    or test_file_path.endswith(f"/{block}")
                ):
                    matched = True
                    break
            if matched:
                selected.append(item)
            else:
                deselected.append(item)
        if deselected:
            items[:] = selected
            config.hook.pytest_deselected(items=deselected)

    if config.getoption("--git-diff"):
        changed_files = set()

        def run_cmd(cmd):
            try:
                res = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, check=True
                )
                return [line.strip() for line in res.stdout.split("\n") if line.strip()]
            except Exception:
                return []

        changed_files.update(run_cmd("git diff --name-only"))
        changed_files.update(run_cmd("git diff --cached --name-only"))

        target = config.getoption("--git-target")
        if not target:
            if run_cmd("git rev-parse --verify origin/main"):
                target = "origin/main"
            elif run_cmd("git rev-parse --verify main"):
                target = "main"
            else:
                target = "HEAD~1"

        if target:
            merge_base = run_cmd(f"git merge-base HEAD {target}")
            if merge_base:
                changed_files.update(run_cmd(f"git diff --name-only {merge_base[0]}"))
            else:
                changed_files.update(run_cmd(f"git diff --name-only {target}"))

        if not changed_files:
            print("\n[pytest-git-diff] No changed files detected. Running all tests.")
            return

        global_impact_files = {
            "pyproject.toml",
            "uv.lock",
            "Makefile",
            "Dockerfile",
            "settings.yaml",
            "settings-test.yaml",
            "tests/conftest.py",
            "private_gpt/di.py",
            "private_gpt/initialize.py",
            "private_gpt/main.py",
            "private_gpt/__main__.py",
            "private_gpt/constants.py",
            "private_gpt/paths.py",
        }

        has_global_change = any(f in global_impact_files for f in changed_files)
        if has_global_change:
            print(
                "\n[pytest-git-diff] Global config or core files changed. Running all tests."
            )
            return

        selected = []
        deselected = []

        for item in items:
            test_file_path = str(
                getattr(item, "path", getattr(item, "fspath", ""))
            ).replace("\\", "/")
            affected = False
            for f in changed_files:
                f = f.replace("\\", "/")
                if f == test_file_path:
                    affected = True
                    break
                if f.startswith("private_gpt/"):
                    subpath = f[len("private_gpt/") :]
                    if "/" not in subpath:
                        affected = True
                        break
                    subpath_dir = "/".join(subpath.split("/")[:-1])
                    if subpath_dir and subpath_dir in test_file_path:
                        affected = True
                        break
                    filename = subpath.split("/")[-1]
                    if filename.endswith(".py"):
                        mod_name = filename[:-3]
                        if f"test_{mod_name}.py" in test_file_path:
                            affected = True
                            break
                if f.startswith("tests/fixtures/") or f.startswith("tests/settings/"):
                    affected = True
                    break
                if f.startswith("tests/") and (
                    f in test_file_path or test_file_path in f
                ):
                    affected = True
                    break

            if affected:
                selected.append(item)
            else:
                deselected.append(item)

        if deselected:
            items[:] = selected
            config.hook.pytest_deselected(items=deselected)
