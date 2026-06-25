"""Unit tests for skills_files.py wrapper directory handling."""

import io
import zipfile

import pytest
from starlette.datastructures import Headers, UploadFile

from private_gpt.components.skills.errors import SkillDomainError, SkillErrorCode
from private_gpt.server.skills.skills_files import (
    _extract_zip,
    _flatten_wrapper_directory,
    stored_files_from_uploads,
)


def _make_upload(payload: bytes, filename: str = "skill.zip") -> UploadFile:
    return UploadFile(
        file=io.BytesIO(payload),
        filename=filename,
        headers=Headers({"content-type": "application/zip"}),
    )


def _zip_with_files(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, content in files.items():
            zf.writestr(path, content)
    return buf.getvalue()


def _empty_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w"):
        pass
    return buf.getvalue()


class TestExtractZip:
    def test_empty_zip_raises_empty_zip(self):
        upload = _make_upload(_empty_zip())
        with pytest.raises(SkillDomainError) as exc_info:
            _extract_zip(upload, _empty_zip())
        assert exc_info.value.code == SkillErrorCode.EMPTY_ZIP

    def test_all_empty_files_raises_empty_zip(self):
        payload = _zip_with_files({"SKILL.md": b"", "scripts/run.py": b""})
        upload = _make_upload(payload)
        with pytest.raises(SkillDomainError) as exc_info:
            _extract_zip(upload, payload)
        assert exc_info.value.code == SkillErrorCode.EMPTY_ZIP

    def test_empty_skill_md_alone_raises_empty_zip(self):
        payload = _zip_with_files({"SKILL.md": b""})
        upload = _make_upload(payload)
        with pytest.raises(SkillDomainError) as exc_info:
            _extract_zip(upload, payload)
        assert exc_info.value.code == SkillErrorCode.EMPTY_ZIP

    def test_non_empty_files_pass_through(self):
        payload = _zip_with_files({"SKILL.md": b"---\nname: x\n---", "run.py": b""})
        upload = _make_upload(payload)
        result = _extract_zip(upload, payload)
        assert any(p == "SKILL.md" for p, _ in result)

    def test_invalid_zip_raises_invalid_zip(self):
        upload = _make_upload(b"not a zip file")
        with pytest.raises(SkillDomainError) as exc_info:
            _extract_zip(upload, b"not a zip file")
        assert exc_info.value.code == SkillErrorCode.INVALID_ZIP


def _create_zip_entries(files: dict[str, bytes]) -> list[tuple[str, bytes]]:
    """Helper to create fake zip entries from a dict."""
    return list(files.items())


class TestFlattenWrapperDirectory:
    """Tests for _flatten_wrapper_directory function."""

    def test_no_wrapper_returns_as_is(self):
        """When SKILL.md is at root, no flattening occurs."""
        entries = [
            ("SKILL.md", b"---\nname: test\ndescription: A test\n---"),
            ("scripts/run.py", b"print('hi')"),
            ("references/README.md", b"# Reference"),
        ]
        result = _flatten_wrapper_directory(entries)
        assert len(result) == 3
        paths = [p for p, _ in result]
        assert "SKILL.md" in paths
        assert "scripts/run.py" in paths

    def test_single_level_wrapper_flattened(self):
        """GitHub-style wrapper (repo-name/SKILL.md) is flattened."""
        entries = [
            (
                "audit-skills-main/SKILL.md",
                b"---\nname: audit\ndescription: Audit skill\n---",
            ),
            ("audit-skills-main/README.md", b"# Audit Skills"),
            ("audit-skills-main/references/GUIDE.md", b"# Guide"),
            ("audit-skills-main/scripts/check.sh", b"#!/bin/bash"),
        ]
        result = _flatten_wrapper_directory(entries)
        paths = [p for p, _ in result]
        assert "SKILL.md" in paths
        assert "README.md" in paths
        assert "references/GUIDE.md" in paths
        assert "scripts/check.sh" in paths
        # Wrapper prefix should be removed
        assert not any(p.startswith("audit-skills-main/") for p in paths)

    def test_nested_wrapper_flattened(self):
        """Nested wrappers (outer/inner/SKILL.md) are fully flattened.

        This simulates ios-simulator-skill-main.zip where:
        - ios-simulator-skill-main/ contains repo-level files
        - ios-simulator-skill-main/ios-simulator-skill/ contains the actual skill
        """
        entries = [
            # Files at outer level (should be EXCLUDED)
            ("ios-simulator-skill-main/README.md", b"# Repo README"),
            ("ios-simulator-skill-main/CLAUDE.md", b"# Claude config"),
            ("ios-simulator-skill-main/.gitignore", b"*build*"),
            # Actual skill files (should be INCLUDED and FLATTENED)
            (
                "ios-simulator-skill-main/ios-simulator-skill/SKILL.md",
                b"---\nname: ios-sim\ndescription: iOS simulator\n---",
            ),
            (
                "ios-simulator-skill-main/ios-simulator-skill/scripts/accessibility_audit.py",
                b"import sys",
            ),
            (
                "ios-simulator-skill-main/ios-simulator-skill/scripts/app_launcher.py",
                b"import os",
            ),
            (
                "ios-simulator-skill-main/ios-simulator-skill/.claude-plugin/plugin.json",
                b'{"name": "test"}',
            ),
        ]
        result = _flatten_wrapper_directory(entries)
        paths = [p for p, _ in result]

        # SKILL.md should be at root
        assert "SKILL.md" in paths

        # Skill files should be included with correct relative paths
        assert "scripts/accessibility_audit.py" in paths
        assert "scripts/app_launcher.py" in paths
        assert ".claude-plugin/plugin.json" in paths

        # Outer-level files should be EXCLUDED
        assert "README.md" not in paths  # The outer README.md
        assert "CLAUDE.md" not in paths
        assert ".gitignore" not in paths

        # No wrapper prefixes should remain
        assert not any("ios-simulator-skill-main" in p for p in paths)
        assert not any("ios-simulator-skill/" in p for p in paths)

    def test_excludes_siblings_outside_skill_root(self):
        """Files outside the skill root directory are excluded."""
        entries = [
            # Sibling files at same level as skill dir (EXCLUDE)
            ("repo-main/docs/GUIDE.md", b"# Guide - exclude"),
            ("repo-main/tools/util.py", b"# Tools - exclude"),
            # Actual skill files (INCLUDE)
            (
                "repo-main/my-skill/SKILL.md",
                b"---\nname: my-skill\ndescription: My skill\n---",
            ),
            ("repo-main/my-skill/scripts/run.py", b"print('run')"),
        ]
        result = _flatten_wrapper_directory(entries)
        paths = [p for p, _ in result]

        assert "SKILL.md" in paths
        assert "scripts/run.py" in paths
        assert "docs/GUIDE.md" not in paths
        assert "tools/util.py" not in paths

    def test_empty_entries(self):
        """Empty entry list returns empty list."""
        result = _flatten_wrapper_directory([])
        assert result == []

    def test_no_skill_md_returns_original(self):
        """If no SKILL.md found, original entries are returned."""
        entries = [
            ("some-dir/README.md", b"# Readme"),
            ("some-dir/data.txt", b"data"),
        ]
        result = _flatten_wrapper_directory(entries)
        assert len(result) == 2
        assert "some-dir/README.md" in [p for p, _ in result]

    def test_case_insensitive_skill_md_detection(self):
        """skill.md (lowercase) is also recognized."""
        entries = [
            ("repo/skill.md", b"---\nname: test\ndescription: Test\n---"),
            ("repo/scripts/run.py", b"print('hi')"),
        ]
        result = _flatten_wrapper_directory(entries)
        paths = [p for p, _ in result]
        # Should normalize to SKILL.md
        assert "SKILL.md" in paths or "skill.md" in paths

    def test_mixed_content_at_multiple_levels(self):
        """Complex structure with files at multiple levels."""
        entries = [
            # Root level (exclude if not part of skill)
            ("project/LICENSE", b"MIT License"),
            ("project/CONTRIBUTING.md", b"# Contributing"),
            # Skill level (include)
            (
                "project/my-skill/SKILL.md",
                b"---\nname: my-skill\ndescription: Skill\n---",
            ),
            ("project/my-skill/README.md", b"# Skill README"),
            ("project/my-skill/scripts/helper.py", b"def help(): pass"),
            ("project/my-skill/references/api.md", b"# API Docs"),
            ("project/my-skill/assets/template.json", b'{"template": true}'),
            # Another sibling dir (exclude)
            (
                "project/other-skill/SKILL.md",
                b"---\nname: other\ndescription: Other\n---",
            ),
        ]
        result = _flatten_wrapper_directory(entries)
        paths = [p for p, _ in result]

        # Only my-skill contents should be included
        assert "SKILL.md" in paths
        assert "README.md" in paths  # From my-skill
        assert "scripts/helper.py" in paths
        assert "references/api.md" in paths
        assert "assets/template.json" in paths

        # Project-level and other-skill files excluded
        assert "LICENSE" not in paths
        assert "CONTRIBUTING.md" not in paths
        assert not any("other-skill" in p for p in paths)


class TestRealWorldScenarios:
    """Tests based on real-world zip structures."""

    def test_audit_skills_structure(self):
        """Simulates audit-skills-main.zip structure."""
        # Real structure from unzip -l output
        entries = {
            "audit-skills-main/.github/workflows/skills-installation-ci.yml": b"name: CI",
            "audit-skills-main/README.md": b"# Audit Skills",
            "audit-skills-main/SKILL.md": b"---\nname: audit-skills\ndescription: Security auditing\n---",
            "audit-skills-main/references/report-template.md": b"# Report Template",
            "audit-skills-main/references/vulnerability-checklist.md": b"# Checklist",
            "audit-skills-main/scripts/ci/check-skill-exploits.sh": b"#!/bin/bash",
            "audit-skills-main/scripts/ci/test-npx-skills-add.sh": b"#!/bin/bash",
            "audit-skills-main/scripts/ci/validate-skills-install.sh": b"#!/bin/bash",
            "audit-skills-main/skills/audit-skills/README.md": b"# Nested README",
            "audit-skills-main/skills/audit-skills/SKILL.md": b"---\nname: nested\ndescription: Nested\n---",
        }
        result = _flatten_wrapper_directory(list(entries.items()))
        paths = sorted([p for p, _ in result])

        # All files should be flattened one level
        assert "SKILL.md" in paths
        assert "README.md" in paths
        assert "references/report-template.md" in paths
        assert "scripts/ci/check-skill-exploits.sh" in paths
        assert ".github/workflows/skills-installation-ci.yml" in paths

    def test_ios_simulator_skill_structure(self):
        """Simulates ios-simulator-skill-main.zip structure.

        This has a nested skill directory inside the repo wrapper.
        """
        entries = {
            # Repo-level files (will be excluded)
            "ios-simulator-skill-main/README.md": b"# iOS Simulator Skill Repo",
            "ios-simulator-skill-main/CLAUDE.md": b"# Claude Config",
            "ios-simulator-skill-main/.gitignore": b"__pycache__/",
            "ios-simulator-skill-main/pyproject.toml": b"[project]",
            # Actual skill files (will be included)
            "ios-simulator-skill-main/ios-simulator-skill/SKILL.md": b"---\nname: ios-simulator-skill\ndescription: Automates iOS simulator\n---",
            "ios-simulator-skill-main/ios-simulator-skill/.claude-plugin/plugin.json": b'{"name": "ios-simulator"}',
            "ios-simulator-skill-main/ios-simulator-skill/scripts/accessibility_audit.py": b"import sys",
            "ios-simulator-skill-main/ios-simulator-skill/scripts/app_launcher.py": b"import os",
            "ios-simulator-skill-main/ios-simulator-skill/scripts/common/__init__.py": b"",
            "ios-simulator-skill-main/ios-simulator-skill/references/accessibility_checklist.md": b"# Checklist",
        }
        result = _flatten_wrapper_directory(list(entries.items()))
        paths = sorted([p for p, _ in result])

        # Skill files should be at root after flattening
        assert "SKILL.md" in paths
        assert ".claude-plugin/plugin.json" in paths
        assert "scripts/accessibility_audit.py" in paths
        assert "scripts/app_launcher.py" in paths
        assert "scripts/common/__init__.py" in paths
        assert "references/accessibility_checklist.md" in paths

        # Repo-level files should be excluded
        assert "README.md" not in paths  # The repo README, not skill README
        assert "CLAUDE.md" not in paths
        assert ".gitignore" not in paths
        assert "pyproject.toml" not in paths

        # No wrapper prefixes
        assert not any("ios-simulator-skill-main" in p for p in paths)
        assert not any("ios-simulator-skill/" in p for p in paths)


class TestStoredFilesFromUploads:
    async def test_single_text_file_promoted_to_skill_md(self):
        content = b"---\nname: my-skill\ndescription: test\n---"
        upload = UploadFile(
            file=io.BytesIO(content),
            filename="my-skill.md",
            headers=Headers({"content-type": "text/markdown"}),
        )
        result = await stored_files_from_uploads([upload])
        assert len(result) == 1
        assert result[0].path == "SKILL.md"
        assert result[0].content == content

    async def test_single_binary_file_not_promoted_to_skill_md(self):
        binary = b"\x50\x4b\x03\x04" + b"\x00" * 20  # zip magic bytes
        upload = UploadFile(
            file=io.BytesIO(binary),
            filename="data.bin",
            headers=Headers({"content-type": "application/octet-stream"}),
        )
        with pytest.raises(SkillDomainError) as exc_info:
            await stored_files_from_uploads([upload])
        assert exc_info.value.code == SkillErrorCode.MISSING_SKILL_MD
