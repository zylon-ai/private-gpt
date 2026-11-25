#!/usr/bin/env python3
"""Validate release-line ownership and forward-port merged fixes without release metadata.

Shared with component repos through zylon-base-template. Uses only Python's
standard library, git and gh; never installs or runs code from the source PR.
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess

POLICY_PATH = ".github/release-policy.json"
LINE_PATH = ".github/release-line.json"


def git(*args, check=True, input=None):
    return subprocess.run(
        ["git", *args], input=input, capture_output=True, check=check
    ).stdout


def gh(*args):
    return json.loads(subprocess.check_output(["gh", *args], text=True))


def version_line(version):
    match = re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version)
    if not match:
        raise ValueError(f"Expected a stable component version, got {version!r}")
    return f"{match[1]}.{match[2]}"


def validate(root, branch):
    root = Path(root)
    policy = json.loads((root / POLICY_PATH).read_text())
    maintenance = branch.startswith("release/")
    if branch != policy["trunk"] and not maintenance:
        raise ValueError(f"{branch} does not own a release version range")
    path = policy["maintenance-config"] if maintenance else policy["config"]
    config = json.loads((root / path).read_text())
    strategy = "always-bump-patch" if maintenance else "always-bump-minor"
    for package in config["packages"].values():
        if package.get("versioning", config.get("versioning")) != strategy:
            raise ValueError(f"{path} must use {strategy}")
        if package.get("release-as") or config.get("release-as"):
            raise ValueError("release-as overrides bypass release-line ownership")
    manifest = json.loads((root / policy["manifest"]).read_text())
    if set(manifest) != {"."} or set(config["packages"]) != {"."}:
        raise ValueError("This policy requires one component version per repository")
    line = version_line(manifest["."])
    if (
        not maintenance
        and manifest["."] != policy["adoption-version"]
        and not manifest["."].endswith(".0")
    ):
        raise ValueError(
            "Trunk publishes new minor versions; patch versions belong to maintenance"
        )
    if maintenance:
        marker = json.loads((root / LINE_PATH).read_text())
        if branch != f"release/{line}" or marker["line"] != line:
            raise ValueError(f"Manifest {manifest['.']} does not belong to {branch}")
    return policy


def split_paths(paths, policy):
    """Discard dedicated metadata; refuse mixed source/version files, never partial fixes."""
    metadata = set(policy["metadata"]) | {POLICY_PATH, LINE_PATH}
    mixed = set(policy.get("mixed-version-files", []))
    blocked = sorted(set(paths) & mixed)
    if blocked:
        raise ValueError(
            "Manual forward-port required for files containing both code/dependencies "
            "and release versions: " + ", ".join(blocked)
        )
    return [
        path
        for path in paths
        if path not in metadata
        and not path.startswith(".github/workflows/")
        and not path.startswith(".github/scripts/")
    ]


def forward_port(event, policy):
    pr = event["pull_request"]
    if not pr.get("merged") or not pr["base"]["ref"].startswith("release/"):
        return
    labels = {label["name"] for label in pr.get("labels", [])}
    if pr["head"]["ref"].startswith("release-please--") or any(
        label.startswith("autorelease:") for label in labels
    ):
        print("Release PR: no code forward-port required.")
        return
    repo = os.environ["GITHUB_REPOSITORY"]
    number = int(pr["number"])
    sha = pr["merge_commit_sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("Invalid merge commit SHA")
    # The checkout and script are from the trusted trunk. Fetch objects only; do not
    # execute source-branch hooks, scripts, actions, dependency installers or tests.
    git(
        "fetch",
        "origin",
        sha,
        f"refs/heads/{policy['trunk']}:refs/remotes/origin/{policy['trunk']}",
    )
    head = f"forwardport/pr-{number}"
    existing = gh(
        "pr",
        "list",
        "--repo",
        repo,
        "--head",
        head,
        "--state",
        "all",
        "--json",
        "number,url",
    )
    if existing:
        print(f"Forward-port already exists: {existing[0]['url']}")
        return
    details = gh("api", f"repos/{repo}/pulls/{number}")
    parent = sha + "^1"
    if (
        details["commits"] > 1
        and len(git("rev-list", "--parents", "-n", "1", sha).split()) == 2
    ):
        # Squash and rebase merges both have one parent. Compare the complete PR
        # patch with the merged commit so a rebase cannot forward only its last fix.
        source_head = details["head"]["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", source_head):
            raise ValueError("Invalid PR head SHA")
        first = gh("api", f"repos/{repo}/pulls/{number}/commits?per_page=1")[0]["sha"]
        if not re.fullmatch(r"[0-9a-f]{40}", first):
            raise ValueError("Invalid first PR commit SHA")
        git("fetch", "origin", source_head, first)
        # A rebase can reuse the original commits. In that case the merge-base
        # would hide all but the last commit, so reject before comparing patches.
        if (
            subprocess.run(
                ["git", "merge-base", "--is-ancestor", first, parent],
                capture_output=True,
                check=False,
            ).returncode
            == 0
        ):
            raise ValueError(
                "Merged commit does not contain the complete PR patch; forward-port manually."
            )
        # Exclude target commits merged into the hotfix before its squash merge.
        base = git("merge-base", parent, source_head).decode().strip()
        whole_pr = git("diff", "--binary", base, source_head)
        merged = git("diff", "--binary", parent, sha)
        if git("patch-id", "--stable", input=whole_pr) != git(
            "patch-id", "--stable", input=merged
        ):
            raise ValueError(
                "Merged commit does not contain the complete PR patch; forward-port manually."
            )
    paths = (
        git("diff", "--name-only", "-z", "--no-renames", parent, sha)
        .decode()
        .split("\0")
    )
    paths = split_paths([p for p in paths if p], policy)
    if not paths:
        print("Only release/automation metadata changed; nothing to forward-port.")
        return
    patch = git(
        "diff",
        "--binary",
        "--full-index",
        "--no-renames",
        parent,
        sha,
        "--",
        *[":(literal)" + path for path in paths],
    )
    git("checkout", "-B", head, f"origin/{policy['trunk']}")
    git("apply", "--3way", "--index", input=patch)
    if not git("diff", "--cached", "--name-only").strip():
        print("Fix is already present on trunk.")
        return
    git("config", "user.name", "github-actions[bot]")
    git("config", "user.email", "41898282+github-actions[bot]@users.noreply.github.com")
    title = f"fix: forward-port #{number} from {pr['base']['ref']}"
    git(
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-m",
        title,
        "-m",
        f"Source-PR: {pr['html_url']}\nSource-Commit: {sha}",
    )
    # Never force-push over human edits. A failed PR creation can be retried from the
    # existing branch; otherwise a differing remote branch requires manual review.
    remote = (
        git("ls-remote", "--heads", "origin", f"refs/heads/{head}").decode().strip()
    )
    if remote:
        git("fetch", "origin", head)
        if git("rev-parse", "FETCH_HEAD^{tree}") != git("rev-parse", "HEAD^{tree}"):
            raise ValueError(
                f"{head} already exists with different changes; inspect it manually"
            )
    else:
        git("push", "origin", f"HEAD:refs/heads/{head}")
    body = (
        f"Forward-ports the code changes from {pr['html_url']} onto `{policy['trunk']}`.\n\n"
        "Release metadata and automation files are excluded. Review and run normal CI before merging.\n"
    )
    subprocess.run(
        [
            "gh",
            "pr",
            "create",
            "--repo",
            repo,
            "--base",
            policy["trunk"],
            "--head",
            head,
            "--title",
            title,
            "--body",
            body,
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["validate", "forward-port"])
    parser.add_argument("--branch", default=os.environ.get("GITHUB_REF_NAME", ""))
    args = parser.parse_args()
    if args.command == "validate":
        validate(".", args.branch)
    else:
        policy = json.loads(Path(POLICY_PATH).read_text())
        forward_port(
            json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()), policy
        )


if __name__ == "__main__":
    main()
