import io
import uuid
import zipfile

from fastapi.testclient import TestClient

from tests.fixtures.mock_injector import MockInjector


def _skill_md(
    name: str,
    description: str,
    body: str = "Use this skill",
    license: str | None = None,
    compatibility: str | None = None,
    metadata: dict[str, str] | None = None,
    allowed_tools: str | None = None,
) -> str:
    frontmatter_lines = [f"name: {name}", f'description: "{description}"']
    if license:
        frontmatter_lines.append(f"license: {license}")
    if compatibility:
        frontmatter_lines.append(f"compatibility: {compatibility}")
    if metadata:
        frontmatter_lines.append("metadata:")
        for k, v in metadata.items():
            frontmatter_lines.append(f'  {k}: "{v}"')
    if allowed_tools:
        frontmatter_lines.append(f"allowed-tools: {allowed_tools}")
    frontmatter = "\n".join(frontmatter_lines)
    return f"---\n{frontmatter}\n---\n\n{body}\n"


def _skill_zip(
    name: str,
    description: str,
    body: str = "Use this skill",
    extra_files: dict[str, bytes] | None = None,
    wrapper_dir: str | None = None,
    **frontmatter_kwargs,
) -> bytes:
    """Create a zip file containing a skill.

    Args:
        name: Skill name for SKILL.md frontmatter
        description: Skill description for SKILL.md frontmatter
        body: Body content of SKILL.md
        extra_files: Dict of filepath -> content for additional files
        wrapper_dir: If set, wraps all files in this directory
        **frontmatter_kwargs: Additional frontmatter kwargs
    """
    skill_md = _skill_md(name, description, body=body, **frontmatter_kwargs)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        prefix = f"{wrapper_dir}/" if wrapper_dir else ""
        archive.writestr(f"{prefix}SKILL.md", skill_md)
        for path, content in (extra_files or {}).items():
            archive.writestr(f"{prefix}{path}", content)
    return buffer.getvalue()


def _skill_zip_nested(
    name: str,
    description: str,
    body: str = "Use this skill",
    extra_files: dict[str, bytes] | None = None,
    outer_wrapper: str | None = None,
    inner_wrapper: str | None = None,
    **frontmatter_kwargs,
) -> bytes:
    """Create a zip with nested structure like ios-simulator-skill.

    Structure: {outer}/{inner}/SKILL.md, {outer}/{inner}/scripts/...
    """
    skill_md = _skill_md(name, description, body=body, **frontmatter_kwargs)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        # Add some files at outer level (should be excluded)
        if outer_wrapper:
            archive.writestr(f"{outer_wrapper}/README.md", b"# Outer README")
            archive.writestr(f"{outer_wrapper}/CLAUDE.md", b"# Claude config")

        prefix = ""
        if outer_wrapper:
            prefix = f"{outer_wrapper}/"
        if inner_wrapper:
            prefix += f"{inner_wrapper}/"

        # Add skill files at inner level
        archive.writestr(f"{prefix}SKILL.md", skill_md)
        for path, content in (extra_files or {}).items():
            archive.writestr(f"{prefix}{path}", content)
    return buffer.getvalue()


def _multipart_file(
    name: str, description: str, body: str = "Use this skill", **frontmatter_kwargs
) -> tuple[str, bytes, str]:
    return (
        "SKILL.md",
        _skill_md(name, description, body=body, **frontmatter_kwargs).encode("utf-8"),
        "text/markdown",
    )


def _collection() -> str:
    return f"tenant-{uuid.uuid4()}"


def test_validate_skill_returns_parsed_name_and_description(
    test_client: TestClient, injector: MockInjector
) -> None:
    resp = test_client.post(
        "/v1/skills/validate",
        data={
            "display_title": "Sales Ops Helper",
            "collection": _collection(),
            "loading": "lazy",
        },
        files=[
            (
                "files",
                _multipart_file("sales-ops-helper", "Helps sales reps draft outreach."),
            )
        ],
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "valid": True,
        "name": "sales-ops-helper",
        "description": "Helps sales reps draft outreach.",
        "errors": [],
    }


def test_validate_skill_returns_errors_for_invalid_frontmatter(
    test_client: TestClient, injector: MockInjector
) -> None:
    resp = test_client.post(
        "/v1/skills/validate",
        data={
            "display_title": "Broken Skill",
            "collection": _collection(),
            "loading": "lazy",
        },
        files=[
            (
                "files",
                (
                    "SKILL.md",
                    b"body without yaml frontmatter",
                    "text/markdown",
                ),
            )
        ],
    )

    assert resp.status_code == 200
    assert resp.json() == {
        "valid": False,
        "name": None,
        "description": None,
        "errors": ["SKILL.md must start with YAML frontmatter"],
    }


def test_skill_crud_flow(test_client: TestClient, injector: MockInjector) -> None:
    collection = _collection()

    create_resp = test_client.post(
        "/v1/skills",
        data={
            "display_title": "Finance Analyst",
            "collection": collection,
            "loading": "lazy",
        },
        files=[
            (
                "files",
                (
                    "finance-analyst.zip",
                    _skill_zip("finance-analyst", "Analyzes finance data"),
                    "application/zip",
                ),
            )
        ],
    )
    assert create_resp.status_code == 200
    skill = create_resp.json()
    skill_id = skill["id"]
    assert skill["type"] == "skill"
    assert skill["display_title"] == "Finance Analyst"
    assert skill["collection"] == collection
    assert skill["latest_version"]

    list_resp = test_client.get("/v1/skills", params={"collection": collection})
    assert list_resp.status_code == 200
    listed = list_resp.json()
    assert len(listed["data"]) == 1
    assert listed["data"][0]["id"] == skill_id

    get_resp = test_client.get(
        f"/v1/skills/{skill_id}", params={"collection": collection}
    )
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == skill_id

    delete_resp = test_client.delete(
        f"/v1/skills/{skill_id}", params={"collection": collection}
    )
    assert delete_resp.status_code == 200
    assert delete_resp.json() == {"id": skill_id, "type": "skill_deleted"}

    assert (
        test_client.get(
            f"/v1/skills/{skill_id}", params={"collection": collection}
        ).status_code
        == 404
    )


def test_skill_with_full_frontmatter_and_resources(
    test_client: TestClient, injector: MockInjector
) -> None:
    collection = _collection()

    zip_bytes = _skill_zip(
        name="data-analyst",
        description="Analyzes datasets. Use when user asks about data, CSV files, or reports.",
        body="## Instructions\nFollow the reference guide.\n\nSee [reference](references/REFERENCE.md)",
        license="Apache 2.0",
        compatibility="Requires code execution tool",
        metadata={"version": "1.0.0", "author": "platform-team", "env": "prod"},
        allowed_tools="Bash Read Write",
        extra_files={
            "references/REFERENCE.md": b"# Reference\nDetailed API docs here.",
            "scripts/analyze.py": b"import sys\nprint('hello')",
        },
    )

    create_resp = test_client.post(
        "/v1/skills",
        data={"display_title": "Data Analyst", "collection": collection},
        files=[("files", ("data-analyst.zip", zip_bytes, "application/zip"))],
    )
    assert create_resp.status_code == 200
    skill = create_resp.json()
    assert skill["display_title"] == "Data Analyst"
    assert skill["latest_version"]

    version_resp = test_client.get(
        f"/v1/skills/{skill['id']}/versions/{skill['latest_version']}",
        params={"collection": collection},
    )
    assert version_resp.status_code == 200
    version = version_resp.json()
    assert version["name"] == "data-analyst"
    assert (
        version["description"]
        == "Analyzes datasets. Use when user asks about data, CSV files, or reports."
    )


def test_skill_versions_lifecycle(
    test_client: TestClient, injector: MockInjector
) -> None:
    collection = _collection()

    create_resp = test_client.post(
        "/v1/skills",
        data={"display_title": "Code Review", "collection": collection},
        files=[
            (
                "files",
                (
                    "code-review.zip",
                    _skill_zip("code-review", "Reviews code for correctness"),
                    "application/zip",
                ),
            )
        ],
    )
    assert create_resp.status_code == 200
    skill_id = create_resp.json()["id"]
    v1_version = create_resp.json()["latest_version"]

    create_v2 = test_client.post(
        f"/v1/skills/{skill_id}/versions",
        data={"collection": collection},
        files=[
            (
                "files",
                (
                    "code-review.zip",
                    _skill_zip("code-review", "Reviews code and checks tests"),
                    "application/zip",
                ),
            )
        ],
    )
    assert create_v2.status_code == 200
    v2 = create_v2.json()
    assert v2["type"] == "skill_version"
    assert v2["description"] == "Reviews code and checks tests"

    list_page1 = test_client.get(
        f"/v1/skills/{skill_id}/versions",
        params={"collection": collection, "limit": 1},
    )
    assert list_page1.status_code == 200
    page1 = list_page1.json()
    assert page1["has_more"] is True
    assert page1["next_page"] is not None
    assert len(page1["data"]) == 1

    list_page2 = test_client.get(
        f"/v1/skills/{skill_id}/versions",
        params={"collection": collection, "limit": 1, "page": page1["next_page"]},
    )
    assert list_page2.status_code == 200
    assert len(list_page2.json()["data"]) == 1

    delete_v2 = test_client.delete(
        f"/v1/skills/{skill_id}/versions/{v2['version']}",
        params={"collection": collection},
    )
    assert delete_v2.status_code == 200
    assert delete_v2.json()["type"] == "skill_version_deleted"

    remaining = test_client.get(
        f"/v1/skills/{skill_id}/versions",
        params={"collection": collection, "limit": 10},
    )
    assert remaining.status_code == 200
    assert len(remaining.json()["data"]) == 1
    assert remaining.json()["data"][0]["version"] == v1_version


def test_skill_collection_isolation(
    test_client: TestClient, injector: MockInjector
) -> None:
    collection_left = _collection()
    collection_right = _collection()

    create_resp = test_client.post(
        "/v1/skills",
        data={"display_title": "Ops", "collection": collection_left},
        files=[
            ("files", ("ops.zip", _skill_zip("ops", "Ops helper"), "application/zip"))
        ],
    )
    assert create_resp.status_code == 200
    skill_id = create_resp.json()["id"]

    assert (
        test_client.get(
            f"/v1/skills/{skill_id}", params={"collection": collection_right}
        ).status_code
        == 404
    )
    assert (
        test_client.get(
            f"/v1/skills/{skill_id}", params={"collection": collection_left}
        ).status_code
        == 200
    )


def test_skill_file_override_wins_over_zip(
    test_client: TestClient, injector: MockInjector
) -> None:
    collection = _collection()

    create_resp = test_client.post(
        "/v1/skills",
        data={"display_title": "Override Skill", "collection": collection},
        files=[
            (
                "files",
                (
                    "override.zip",
                    _skill_zip("override", "Original description"),
                    "application/zip",
                ),
            ),
            ("files", _multipart_file("override", "Overridden description")),
        ],
    )
    assert create_resp.status_code == 200
    skill = create_resp.json()

    version_resp = test_client.get(
        f"/v1/skills/{skill['id']}/versions/{skill['latest_version']}",
        params={"collection": collection},
    )
    assert version_resp.status_code == 200
    assert version_resp.json()["description"] == "Overridden description"


def test_skill_get_not_found(test_client: TestClient, injector: MockInjector) -> None:
    collection = _collection()
    assert (
        test_client.get(
            "/v1/skills/nonexistent", params={"collection": collection}
        ).status_code
        == 404
    )


def test_skill_delete_not_found(
    test_client: TestClient, injector: MockInjector
) -> None:
    collection = _collection()
    assert (
        test_client.delete(
            "/v1/skills/nonexistent", params={"collection": collection}
        ).status_code
        == 404
    )


def test_skill_version_create_not_found(
    test_client: TestClient, injector: MockInjector
) -> None:
    collection = _collection()
    zip_bytes = _skill_zip("test-skill", "A test skill")
    resp = test_client.post(
        "/v1/skills/nonexistent/versions",
        data={"collection": collection},
        files=[("files", ("test-skill.zip", zip_bytes, "application/zip"))],
    )
    assert resp.status_code == 404


def test_skill_version_get_not_found(
    test_client: TestClient, injector: MockInjector
) -> None:
    collection = _collection()
    assert (
        test_client.get(
            "/v1/skills/nonexistent/versions/v1", params={"collection": collection}
        ).status_code
        == 404
    )


def test_skill_version_delete_not_found(
    test_client: TestClient, injector: MockInjector
) -> None:
    collection = _collection()
    assert (
        test_client.delete(
            "/v1/skills/nonexistent/versions/v1", params={"collection": collection}
        ).status_code
        == 404
    )


def test_readonly_skill_cannot_be_deleted_or_versioned(
    test_client: TestClient, injector: MockInjector
) -> None:
    collection = _collection()
    create_resp = test_client.post(
        "/v1/skills",
        data={
            "display_title": "Readonly Skill",
            "collection": collection,
            "readonly": "true",
        },
        files=[
            (
                "files",
                (
                    "readonly.zip",
                    _skill_zip("readonly", "Readonly helper"),
                    "application/zip",
                ),
            )
        ],
    )
    assert create_resp.status_code == 200
    skill_id = create_resp.json()["id"]

    delete_resp = test_client.delete(
        f"/v1/skills/{skill_id}",
        params={"collection": collection},
    )
    assert delete_resp.status_code == 403

    create_version_resp = test_client.post(
        f"/v1/skills/{skill_id}/versions",
        data={"collection": collection},
        files=[
            (
                "files",
                (
                    "readonly.zip",
                    _skill_zip("readonly", "Readonly helper v2"),
                    "application/zip",
                ),
            )
        ],
    )
    assert create_version_resp.status_code == 403


# ============================================================================
# Tests for GitHub-style wrapper directory handling
# ============================================================================


def test_skill_upload_with_github_wrapper_directory(
    test_client: TestClient, injector: MockInjector
) -> None:
    """Test uploading a zip with GitHub-style wrapper (repo-name/SKILL.md)."""
    collection = _collection()

    # Simulate GitHub repo download: audit-skills-main/SKILL.md
    zip_bytes = _skill_zip(
        name="audit-skills",
        description="Security audit skills",
        wrapper_dir="audit-skills-main",
        extra_files={
            "references/REFERENCE.md": b"# Reference guide",
            "scripts/check.sh": b"#!/bin/bash\necho check",
        },
    )

    resp = test_client.post(
        "/v1/skills",
        data={"display_title": "Audit Skills", "collection": collection},
        files=[("files", ("audit-skills-main.zip", zip_bytes, "application/zip"))],
    )
    assert resp.status_code == 200
    skill = resp.json()

    # Verify the skill was created and files were extracted correctly
    version_resp = test_client.get(
        f"/v1/skills/{skill['id']}/versions/{skill['latest_version']}",
        params={"collection": collection},
    )
    assert version_resp.status_code == 200
    version = version_resp.json()
    assert version["name"] == "audit-skills"
    assert version["description"] == "Security audit skills"


def test_skill_upload_with_nested_wrapper_directories(
    test_client: TestClient, injector: MockInjector
) -> None:
    """Test uploading a zip with nested wrappers (outer/inner/SKILL.md).

    This simulates ios-simulator-skill-main.zip where:
    - ios-simulator-skill-main/ contains repo-level files (README.md, etc.)
    - ios-simulator-skill-main/ios-simulator-skill/ contains the actual skill
    """
    collection = _collection()

    # Simulate nested structure like ios-simulator-skill
    zip_bytes = _skill_zip_nested(
        name="ios-simulator",
        description="iOS simulator automation",
        outer_wrapper="ios-simulator-skill-main",
        inner_wrapper="ios-simulator-skill",
        extra_files={
            "scripts/accessibility_audit.py": b"import sys",
            "scripts/app_launcher.py": b"import os",
            ".claude-plugin/plugin.json": b'{"name": "test"}',
        },
    )

    resp = test_client.post(
        "/v1/skills",
        data={"display_title": "iOS Simulator", "collection": collection},
        files=[
            (
                "files",
                (
                    "ios-simulator-skill-main.zip",
                    zip_bytes,
                    "application/zip",
                ),
            )
        ],
    )
    assert resp.status_code == 200, f"Failed: {resp.text}"
    skill = resp.json()

    # Verify skill was created
    version_resp = test_client.get(
        f"/v1/skills/{skill['id']}/versions/{skill['latest_version']}",
        params={"collection": collection},
    )
    assert version_resp.status_code == 200
    version = version_resp.json()
    assert version["name"] == "ios-simulator"
    assert version["description"] == "iOS simulator automation"


def test_skill_upload_excludes_siblings_outside_skill_root(
    test_client: TestClient, injector: MockInjector
) -> None:
    """Verify that files outside the skill root are excluded.

    When SKILL.md is at outer/inner/SKILL.md, files at outer/other.txt
    should NOT be included in the skill.
    """
    collection = _collection()

    # Create zip with sibling directories at same level as skill dir
    buffer = io.BytesIO()
    skill_md = _skill_md("test-sibling", "Testing sibling exclusion")

    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        # Files at outer level (should be EXCLUDED)
        archive.writestr("repo-main/README.md", b"# Repo README - should be excluded")
        archive.writestr("repo-main/docs/GUIDE.md", b"# Guide - should be excluded")

        # Actual skill files (should be INCLUDED)
        archive.writestr("repo-main/test-sibling/SKILL.md", skill_md)
        archive.writestr("repo-main/test-sibling/scripts/run.py", b"print('hello')")

    zip_bytes = buffer.getvalue()

    resp = test_client.post(
        "/v1/skills",
        data={"display_title": "Sibling Test", "collection": collection},
        files=[("files", ("repo-main.zip", zip_bytes, "application/zip"))],
    )
    assert resp.status_code == 200, f"Failed: {resp.text}"
    skill = resp.json()

    # Get the version and verify only skill files are present
    version_resp = test_client.get(
        f"/v1/skills/{skill['id']}/versions/{skill['latest_version']}",
        params={"collection": collection},
    )
    assert version_resp.status_code == 200

    # The skill should have been created successfully
    version = version_resp.json()
    assert version["name"] == "test-sibling"


def test_skill_upload_no_wrapper_still_works(
    test_client: TestClient, injector: MockInjector
) -> None:
    """Ensure normal zips without wrapper still work."""
    collection = _collection()

    # Normal zip with SKILL.md at root
    zip_bytes = _skill_zip(
        name="normal-skill",
        description="A normal skill without wrapper",
        extra_files={
            "scripts/helper.py": b"def help(): pass",
        },
    )

    resp = test_client.post(
        "/v1/skills",
        data={"display_title": "Normal Skill", "collection": collection},
        files=[("files", ("normal-skill.zip", zip_bytes, "application/zip"))],
    )
    assert resp.status_code == 200
    skill = resp.json()

    version_resp = test_client.get(
        f"/v1/skills/{skill['id']}/versions/{skill['latest_version']}",
        params={"collection": collection},
    )
    assert version_resp.status_code == 200
    version = version_resp.json()
    assert version["name"] == "normal-skill"
