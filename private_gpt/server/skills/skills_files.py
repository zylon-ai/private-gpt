import io
import mimetypes
import zipfile
from io import BytesIO
from pathlib import PurePosixPath

from starlette.datastructures import Headers, UploadFile

from private_gpt.components.skills.errors import SkillDomainError, SkillErrorCode
from private_gpt.components.storage.models import StoredFile


async def uploads_from_request_form(
    form_items: list[tuple[str, object]]
) -> list[UploadFile]:
    uploads: list[UploadFile] = []
    for k, v in form_items:
        if k not in {"files", "files[]", "file"}:
            continue
        if isinstance(v, UploadFile):
            uploads.append(v)
        else:
            raw = (
                v
                if isinstance(v, bytes)
                else (v.encode("latin-1") if isinstance(v, str) else None)
            )
            if raw is None:
                continue
            filename, content_type = _infer_file_meta(k, raw)
            uploads.append(
                UploadFile(
                    file=BytesIO(raw),
                    filename=filename,
                    headers=Headers({"content-type": content_type}),
                )
            )
    return uploads


async def stored_files_from_uploads(
    uploads: list[UploadFile],
    default_mime_type: str | None = "application/octet-stream",
) -> list[StoredFile]:
    if not uploads:
        raise SkillDomainError(
            SkillErrorCode.MISSING_FILES, "At least one file is required"
        )

    resolved: dict[str, StoredFile] = {}
    for upload in uploads:
        payload = await upload.read()

        if _is_zip(upload):
            for path, content in _extract_zip(upload, payload):
                normalized = _normalize_path(path)
                resolved[normalized] = StoredFile(
                    path=normalized, content=content, mime_type=None
                )
        else:
            name = _normalize_path(upload.filename or "")
            resolved[name] = StoredFile(
                path=name, content=payload, mime_type=upload.content_type
            )

    if len(resolved) == 1 and "SKILL.md" not in resolved:
        stored = next(iter(resolved.values()))
        resolved = {
            "SKILL.md": StoredFile(
                path="SKILL.md",
                content=stored.content,
                mime_type="text/markdown",
            )
        }

    for value in resolved.values():
        if not value.mime_type:
            mime_type, _ = (
                ("text/markdown", None)
                if value.path.endswith(".md")
                else mimetypes.guess_type(value.path)
            )
            value.mime_type = mime_type or default_mime_type

        if not value.mime_type:
            raise SkillDomainError(
                SkillErrorCode.MIME_TYPE_UNKNOWN,
                f"Could not determine MIME type for file: {value.path}",
            )

    if "SKILL.md" not in resolved:
        raise SkillDomainError(
            SkillErrorCode.MISSING_SKILL_MD, "Upload must include a SKILL.md file."
        )

    return list(resolved.values())


def _normalize_path(path: str) -> str:
    # Normalize backslashes to forward slashes
    path = path.replace("\\", "/")

    parsed = PurePosixPath(path)

    if parsed.is_absolute():
        raise SkillDomainError(
            SkillErrorCode.UNSAFE_PATH_ABSOLUTE,
            f"Unsafe file path (absolute): {path!r}",
        )

    if ".." in parsed.parts:
        raise SkillDomainError(
            SkillErrorCode.UNSAFE_PATH_TRAVERSAL,
            f"Unsafe file path (path traversal): {path!r}",
        )

    parts = list(parsed.parts)
    if parts and parts[-1].lower() == "skill.md":
        parts[-1] = "SKILL.md"
    return "/".join(parts)


_DEFAULT_MIME_TYPE = "application/octet-stream"


def _infer_file_meta(field_name: str, content: bytes | None) -> tuple[str, str]:
    if content is None:
        return field_name, _DEFAULT_MIME_TYPE

    from private_gpt.utils.mime import is_magic_available

    if not is_magic_available():
        return field_name.rstrip("[]"), _DEFAULT_MIME_TYPE

    import magic

    mime = magic.from_buffer(content, mime=True)

    if not isinstance(mime, str) or not mime:
        return field_name.rstrip("[]"), _DEFAULT_MIME_TYPE

    ext = mimetypes.guess_extension(mime) or ""
    filename = f"{field_name.rstrip('[]')}{ext}"
    return filename, mime


def _is_zip(upload: UploadFile) -> bool:
    filename = (upload.filename or "").lower()
    content_type = (upload.content_type or "").lower()
    return filename.endswith(".zip") or content_type in {
        "application/zip",
        "application/x-zip-compressed",
    }


def _extract_zip(upload: UploadFile, payload: bytes) -> list[tuple[str, bytes]]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if not members:
                raise SkillDomainError(
                    SkillErrorCode.EMPTY_ZIP, "ZIP file cannot be empty"
                )

            raw_entries = [
                (item.filename, archive.read(item.filename)) for item in members
            ]
            return _flatten_wrapper_directory(raw_entries)
    except zipfile.BadZipFile as exc:
        raise SkillDomainError(
            SkillErrorCode.INVALID_ZIP,
            f"Invalid ZIP file: {upload.filename or 'unnamed.zip'}",
        ) from exc


def _flatten_wrapper_directory(
    entries: list[tuple[str, bytes]]
) -> list[tuple[str, bytes]]:
    """Find SKILL.md and strip wrapper directories above it.

    Per the Agent Skills spec, SKILL.md defines the skill root.
    This finds SKILL.md, determines its parent directory, and keeps only
    files at or below that level, stripping any wrapper directories above.

    Examples:
      - repo-name/SKILL.md -> SKILL.md
      - repo-name/skill-dir/SKILL.md -> SKILL.md (with sibling dirs excluded)
    """
    if not entries:
        return entries

    # Find SKILL.md (case-insensitive)
    skill_md_entry = None
    for path, content in entries:
        if (
            path.endswith("/SKILL.md")
            or path.endswith("/skill.md")
            or path == "SKILL.md"
            or path == "skill.md"
        ):
            skill_md_entry = (path, content)
            break

    if skill_md_entry is None:
        # No SKILL.md found, return original entries
        return entries

    skill_md_path = skill_md_entry[0]

    # Determine the skill root directory (parent of SKILL.md)
    # e.g., "repo-name/skill-dir/SKILL.md" -> skill_root = "repo-name/skill-dir"
    skill_root = skill_md_path.rsplit("/", 1)[0] if "/" in skill_md_path else ""

    # Filter to only include files at or below the skill root
    filtered = []
    for path, content in entries:
        if skill_root == "":
            # SKILL.md is at root, keep everything
            filtered.append((path, content))
        elif path.startswith(skill_root + "/") or path == skill_root:
            # File is at or below skill root
            filtered.append((path, content))
        # Else: file is outside skill root, exclude it

    # Strip the skill_root prefix from all paths
    result = []
    for path, content in filtered:
        new_path = path[len(skill_root) + 1 :] if skill_root else path
        result.append((new_path, content))

    return result
