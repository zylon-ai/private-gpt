from typing import Annotated, Literal

import base64

from fastapi import APIRouter, Depends, Form, Header, HTTPException, Query, Request
from fastapi.responses import Response

from private_gpt.components.skills.errors import (
    SkillDomainError,
    SkillErrorCode,
    SkillValidationErrors,
)
from private_gpt.components.skills.models.skill_entities import (
    SkillEntity,
    SkillVersionEntity,
)
from private_gpt.components.skills.services.skill_service import SkillService
from private_gpt.server.skills.skill_models import (
    ListSkillsResponse,
    ListSkillVersionFilesResponse,
    ListSkillVersionsResponse,
    SkillDeletedResponse,
    SkillResponse,
    SkillValidationError,
    SkillValidationResponse,
    SkillVersionDeletedResponse,
    SkillVersionFileResponse,
    SkillVersionResponse,
)
from private_gpt.server.skills.skills_files import (
    stored_files_from_uploads,
    uploads_from_request_form,
)
from private_gpt.server.utils.auth import authenticated

skill_router = APIRouter(
    prefix="/v1/skills",
    dependencies=[Depends(authenticated)],
    tags=["Skills"],
    responses={401: {"description": "Unauthorized"}},
)


@skill_router.post(
    "",
    response_model=SkillResponse,
    description="Create a new skill in a collection from multipart form fields and uploaded files.",
    responses={
        200: {
            "description": "Skill created successfully.",
            "content": {
                "application/json": {
                    "examples": {
                        "created_skill": {
                            "summary": "Created custom skill",
                            "value": {
                                "id": "skill_01HZX8N83J4WQ2E4K4G3Q2H9E8",
                                "created_at": "2026-04-16T10:30:00Z",
                                "display_title": "Sales Ops Helper",
                                "latest_version": "v1",
                                "source": "custom",
                                "type": "skill",
                                "updated_at": "2026-04-16T10:30:00Z",
                                "collection": "acme-prod",
                                "loading": "lazy",
                                "readonly": False,
                            },
                        }
                    }
                }
            },
        }
    },
    openapi_extra={
        "requestBody": {
            "description": "Multipart form payload used to create a skill and its initial files.",
            "content": {
                "multipart/form-data": {
                    "examples": {
                        "basic_skill_create": {
                            "summary": "Create skill with collection and title",
                            "value": {
                                "display_title": "Sales Ops Helper",
                                "collection": "acme-prod",
                                "source": "custom",
                                "loading": "lazy",
                                "readonly": False,
                            },
                        }
                    }
                }
            },
        }
    },
)
async def create_skill(
    request: Request,
    display_title: Annotated[
        str,
        Form(
            min_length=1,
            max_length=255,
            examples=["Sales Ops Helper"],
        ),
    ],
    collection: Annotated[
        str,
        Form(
            min_length=1,
            max_length=255,
            examples=["acme-prod"],
        ),
    ],
    source: Annotated[
        Literal["custom", "anthropic", "zylon"],
        Form(examples=["custom"]),
    ] = "custom",
    loading: Annotated[
        Literal["eager", "lazy"],
        Form(examples=["lazy"]),
    ] = "lazy",
    readonly: Annotated[
        bool,
        Form(examples=[False]),
    ] = False,
    anthropic_beta: Annotated[list[str] | None, Header(alias="anthropic-beta")] = None,
) -> SkillResponse:
    """Create a skill and persist its initial version content."""
    del anthropic_beta
    service: SkillService = request.state.injector.get(SkillService)
    form = await request.form()
    uploads = await uploads_from_request_form(list(form.multi_items()))
    files = await stored_files_from_uploads(uploads)
    created = await service.create_skill(
        collection=collection,
        display_title=display_title,
        source=source,
        loading=loading,
        readonly=readonly,
        files=files,
    )
    return _skill_response(created)


@skill_router.get(
    "",
    response_model=ListSkillsResponse,
    description="List skills for a collection with cursor pagination.",
    responses={
        200: {
            "description": "Paginated list of skills.",
            "content": {
                "application/json": {
                    "examples": {
                        "skills_page": {
                            "summary": "First page of skills",
                            "value": {
                                "data": [
                                    {
                                        "id": "skill_01HZX8N83J4WQ2E4K4G3Q2H9E8",
                                        "created_at": "2026-04-16T10:30:00Z",
                                        "display_title": "Sales Ops Helper",
                                        "latest_version": "v3",
                                        "source": "custom",
                                        "type": "skill",
                                        "updated_at": "2026-04-16T12:00:00Z",
                                        "collection": "acme-prod",
                                        "loading": "lazy",
                                        "readonly": False,
                                    }
                                ],
                                "has_more": False,
                                "next_page": None,
                            },
                        }
                    }
                }
            },
        }
    },
)
async def list_skills(
    request: Request,
    collection: str = Query(
        ...,
        min_length=1,
        max_length=255,
        examples=["acme-prod"],
    ),
    limit: int = Query(default=20, ge=1, le=1000, examples=[20]),
    page: str | None = Query(default=None, examples=["eyJvZmZzZXQiOjIwfQ=="]),
    anthropic_beta: Annotated[list[str] | None, Header(alias="anthropic-beta")] = None,
) -> ListSkillsResponse:
    """Return paginated skill metadata for a collection."""
    del anthropic_beta
    service: SkillService = request.state.injector.get(SkillService)
    result = await service.list_skills(collection=collection, limit=limit, page=page)
    return ListSkillsResponse(
        data=[_skill_response(item) for item in result.data],
        has_more=result.has_more,
        next_page=result.next_page,
    )


@skill_router.post(
    "/validate",
    response_model=SkillValidationResponse,
    description="Dry-run validation of skill files and metadata without persisting.",
    responses={
        200: {
            "description": "Validation result.",
            "content": {
                "application/json": {
                    "examples": {
                        "valid": {
                            "summary": "Valid skill",
                            "value": {
                                "valid": True,
                                "name": "sales-ops-helper",
                                "description": "Helps sales reps draft outreach.",
                                "errors": [],
                            },
                        },
                        "invalid": {
                            "summary": "Invalid SKILL.md",
                            "value": {
                                "valid": False,
                                "name": None,
                                "description": None,
                                "errors": [
                                    {
                                        "code": "INVALID_FRONTMATTER",
                                        "message": "SKILL.md must start with YAML frontmatter",
                                    }
                                ],
                            },
                        },
                    }
                }
            },
        }
    },
    openapi_extra={
        "requestBody": {
            "description": "Multipart form payload to validate before creating a skill.",
            "content": {
                "multipart/form-data": {
                    "examples": {
                        "validate_skill": {
                            "summary": "Validate skill with collection and title",
                            "value": {
                                "display_title": "Sales Ops Helper",
                                "collection": "acme-prod",
                                "loading": "lazy",
                            },
                        }
                    }
                }
            },
        }
    },
)
async def validate_skill(
    request: Request,
    display_title: Annotated[
        str,
        Form(
            min_length=1,
            max_length=255,
            examples=["Sales Ops Helper"],
        ),
    ],
    collection: Annotated[
        str,
        Form(
            min_length=1,
            max_length=255,
            examples=["acme-prod"],
        ),
    ],
    loading: Annotated[
        Literal["eager", "lazy"],
        Form(examples=["lazy"]),
    ] = "lazy",
) -> SkillValidationResponse:
    """Validate skill payload (form fields + SKILL.md) without creating anything."""
    del display_title, collection, loading
    service: SkillService = request.state.injector.get(SkillService)
    form = await request.form()
    uploads = await uploads_from_request_form(list(form.multi_items()))

    try:
        files = await stored_files_from_uploads(uploads)
        parsed = await service.validate_skill(files)
    except SkillValidationErrors as exc:
        return SkillValidationResponse(
            valid=False,
            errors=[
                SkillValidationError(
                    code=str(e.code), message=e.message, params=e.params
                )
                for e in exc.errors
            ],
        )
    except SkillDomainError as exc:
        return SkillValidationResponse(
            valid=False,
            errors=[
                SkillValidationError(
                    code=str(exc.code), message=exc.message, params=exc.params
                )
            ],
        )

    return SkillValidationResponse(
        valid=True,
        name=parsed.frontmatter.name,
        description=parsed.frontmatter.description,
    )


@skill_router.get(
    "/{skill_id}",
    response_model=SkillResponse,
    description="Retrieve a single skill by identifier within a collection.",
    responses={
        200: {
            "description": "Skill found.",
            "content": {
                "application/json": {
                    "examples": {
                        "skill": {
                            "summary": "Single skill",
                            "value": {
                                "id": "skill_01HZX8N83J4WQ2E4K4G3Q2H9E8",
                                "created_at": "2026-04-16T10:30:00Z",
                                "display_title": "Sales Ops Helper",
                                "latest_version": "v3",
                                "source": "custom",
                                "type": "skill",
                                "updated_at": "2026-04-16T12:00:00Z",
                                "collection": "acme-prod",
                                "loading": "lazy",
                                "readonly": False,
                            },
                        }
                    }
                }
            },
        }
    },
)
async def get_skill(
    request: Request,
    skill_id: str,
    collection: str = Query(
        ...,
        min_length=1,
        max_length=255,
        examples=["acme-prod"],
    ),
    anthropic_beta: Annotated[list[str] | None, Header(alias="anthropic-beta")] = None,
) -> SkillResponse:
    """Get a skill by id and collection boundary."""
    del anthropic_beta
    service: SkillService = request.state.injector.get(SkillService)
    skill = await service.get_skill(skill_id=skill_id, collection=collection)
    if not skill:
        raise HTTPException(
            status_code=404,
            detail=f"Skill with id {skill_id} not found in collection {collection}",
        )
    return _skill_response(skill)


@skill_router.delete(
    "/{skill_id}",
    response_model=SkillDeletedResponse,
    description="Delete a non-readonly skill from a collection.",
    responses={
        200: {
            "description": "Skill deleted.",
            "content": {
                "application/json": {
                    "examples": {
                        "deleted_skill": {
                            "summary": "Deletion marker",
                            "value": {
                                "id": "skill_01HZX8N83J4WQ2E4K4G3Q2H9E8",
                                "type": "skill_deleted",
                            },
                        }
                    }
                }
            },
        }
    },
)
async def delete_skill(
    request: Request,
    skill_id: str,
    collection: str = Query(
        ...,
        min_length=1,
        max_length=255,
        examples=["acme-prod"],
    ),
    anthropic_beta: Annotated[list[str] | None, Header(alias="anthropic-beta")] = None,
) -> SkillDeletedResponse:
    """Delete a skill and return a deletion marker payload."""
    del anthropic_beta
    service: SkillService = request.state.injector.get(SkillService)
    skill = await service.get_skill(skill_id=skill_id, collection=collection)
    if not skill:
        raise HTTPException(
            status_code=404,
            detail=f"Skill with id {skill_id} not found in collection {collection}",
        )
    if skill.readonly:
        raise HTTPException(
            status_code=403,
            detail=f"Skill with id {skill_id} is readonly and cannot be deleted",
        )
    await service.delete_skill(skill_id=skill_id, collection=collection)
    return SkillDeletedResponse(id=skill_id)


@skill_router.post(
    "/{skill_id}/versions",
    response_model=SkillVersionResponse,
    description="Create a new version for an existing skill using multipart uploads.",
    responses={
        200: {
            "description": "Skill version created successfully.",
            "content": {
                "application/json": {
                    "examples": {
                        "created_version": {
                            "summary": "New skill version",
                            "value": {
                                "id": "version_01HZX8V5AKMFK7YMRQ7QNXMW9B",
                                "created_at": "2026-04-16T12:15:00Z",
                                "description": "Helps sales reps draft concise outreach.",
                                "directory": "sales_ops_helper",
                                "name": "sales_ops_helper",
                                "skill_id": "skill_01HZX8N83J4WQ2E4K4G3Q2H9E8",
                                "type": "skill_version",
                                "version": "v4",
                            },
                        }
                    }
                }
            },
        }
    },
    openapi_extra={
        "requestBody": {
            "description": "Multipart form payload used to create a new skill version.",
            "content": {
                "multipart/form-data": {
                    "examples": {
                        "new_version": {
                            "summary": "Create version in a collection",
                            "value": {
                                "collection": "acme-prod",
                            },
                        }
                    }
                }
            },
        }
    },
)
async def create_skill_version(
    request: Request,
    skill_id: str,
    collection: Annotated[
        str,
        Form(min_length=1, max_length=255, examples=["acme-prod"]),
    ],
    anthropic_beta: Annotated[list[str] | None, Header(alias="anthropic-beta")] = None,
) -> SkillVersionResponse:
    """Create and return a new version for the requested skill."""
    del anthropic_beta
    service: SkillService = request.state.injector.get(SkillService)
    skill = await service.get_skill(skill_id=skill_id, collection=collection)
    if not skill:
        raise HTTPException(
            status_code=404,
            detail=f"Skill with id {skill_id} not found in collection {collection}",
        )
    if skill.readonly:
        raise HTTPException(
            status_code=403,
            detail=f"Skill with id {skill_id} is readonly and cannot create versions",
        )

    form = await request.form()
    uploads = await uploads_from_request_form(list(form.multi_items()))
    files = await stored_files_from_uploads(uploads)

    version = await service.create_version(
        skill_id=skill_id,
        collection=collection,
        files=files,
    )
    if not version:
        raise HTTPException(
            status_code=404,
            detail=f"Skill with id {skill_id} not found in collection {collection}",
        )
    return _version_response(version)


@skill_router.get(
    "/{skill_id}/versions",
    response_model=ListSkillVersionsResponse,
    description="List versions for a skill with cursor pagination.",
    responses={
        200: {
            "description": "Paginated list of skill versions.",
            "content": {
                "application/json": {
                    "examples": {
                        "versions_page": {
                            "summary": "First page of versions",
                            "value": {
                                "data": [
                                    {
                                        "id": "version_01HZX8V5AKMFK7YMRQ7QNXMW9B",
                                        "created_at": "2026-04-16T12:15:00Z",
                                        "description": "Helps sales reps draft concise outreach.",
                                        "directory": "sales_ops_helper",
                                        "name": "sales_ops_helper",
                                        "skill_id": "skill_01HZX8N83J4WQ2E4K4G3Q2H9E8",
                                        "type": "skill_version",
                                        "version": "v4",
                                    }
                                ],
                                "has_more": False,
                                "next_page": None,
                            },
                        }
                    }
                }
            },
        }
    },
)
async def list_skill_versions(
    request: Request,
    skill_id: str,
    collection: str = Query(
        ...,
        min_length=1,
        max_length=255,
        examples=["acme-prod"],
    ),
    limit: int = Query(default=20, ge=1, le=1000, examples=[20]),
    page: str | None = Query(default=None, examples=["eyJvZmZzZXQiOjIwfQ=="]),
    anthropic_beta: Annotated[list[str] | None, Header(alias="anthropic-beta")] = None,
) -> ListSkillVersionsResponse:
    """Return paginated versions for a given skill."""
    del anthropic_beta
    service: SkillService = request.state.injector.get(SkillService)
    skill = await service.get_skill(skill_id=skill_id, collection=collection)
    if not skill:
        raise HTTPException(
            status_code=404,
            detail=f"Skill with id {skill_id} not found in collection {collection}",
        )

    result = await service.list_versions(
        skill_id=skill_id, collection=collection, limit=limit, page=page
    )
    return ListSkillVersionsResponse(
        data=[_version_response(v) for v in result.data],
        has_more=result.has_more,
        next_page=result.next_page,
    )


@skill_router.get(
    "/{skill_id}/versions/{version}/files/{file_path:path}/content",
    summary="Download a skill version file",
    description=(
        "Download the raw bytes of a single file from a skill version bundle. "
        "Paths are relative to the skill root (e.g. `SKILL.md` or `scripts/run.py`)."
    ),
    responses={
        200: {
            "description": "Raw file bytes.",
            "content": {"application/octet-stream": {}},
        },
        400: {"description": "Invalid file path."},
        404: {"description": "Skill, version, or file not found."},
    },
)
async def get_skill_version_file_content(
    request: Request,
    skill_id: str,
    version: str,
    file_path: str,
    collection: str = Query(
        ...,
        min_length=1,
        max_length=255,
        examples=["acme-prod"],
    ),
    anthropic_beta: Annotated[list[str] | None, Header(alias="anthropic-beta")] = None,
) -> Response:
    """Return raw bytes for one file in a skill version."""
    del anthropic_beta
    skill_version = await _require_skill_version(
        request, skill_id=skill_id, version=version, collection=collection
    )
    service: SkillService = request.state.injector.get(SkillService)
    try:
        stored = await service.read_version_file(skill_version, file_path)
    except SkillDomainError as exc:
        status = (
            404 if exc.code == SkillErrorCode.FILE_NOT_FOUND else 400
        )
        raise HTTPException(status_code=status, detail=exc.message) from exc

    filename = stored.path.rsplit("/", 1)[-1]
    return Response(
        content=stored.content,
        media_type=stored.mime_type or "application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@skill_router.get(
    "/{skill_id}/versions/{version}/files",
    response_model=ListSkillVersionFilesResponse,
    summary="List files in a skill version",
    description=(
        "List all files stored for a skill version. "
        "Set `include_content=true` (default) to receive base64-encoded file bodies."
    ),
    responses={
        200: {
            "description": "Skill version files.",
            "content": {
                "application/json": {
                    "examples": {
                        "with_content": {
                            "summary": "Files with content",
                            "value": {
                                "data": [
                                    {
                                        "path": "SKILL.md",
                                        "size_bytes": 120,
                                        "mime_type": "text/markdown",
                                        "content_base64": "LS0tCm5hbWU6IGV4YW1wbGUK...",
                                    },
                                    {
                                        "path": "scripts/run.py",
                                        "size_bytes": 32,
                                        "mime_type": "text/x-python",
                                        "content_base64": "cHJpbnQoJ2hlbGxvJykK",
                                    },
                                ]
                            },
                        }
                    }
                }
            },
        },
        404: {"description": "Skill or version not found."},
    },
)
async def list_skill_version_files(
    request: Request,
    skill_id: str,
    version: str,
    collection: str = Query(
        ...,
        min_length=1,
        max_length=255,
        examples=["acme-prod"],
    ),
    include_content: bool = Query(
        default=True,
        description="When true, each file includes a content_base64 field.",
    ),
    anthropic_beta: Annotated[list[str] | None, Header(alias="anthropic-beta")] = None,
) -> ListSkillVersionFilesResponse:
    """List (and optionally return) all files for a skill version."""
    del anthropic_beta
    skill_version = await _require_skill_version(
        request, skill_id=skill_id, version=version, collection=collection
    )
    service: SkillService = request.state.injector.get(SkillService)
    files = await service.get_version_files(
        skill_version, include_content=include_content
    )
    return ListSkillVersionFilesResponse(
        data=[
            SkillVersionFileResponse(
                path=path,
                size_bytes=size_bytes,
                mime_type=mime_type,
                content_base64=(
                    base64.b64encode(content).decode("ascii")
                    if content is not None
                    else None
                ),
            )
            for path, size_bytes, mime_type, content in files
        ]
    )


@skill_router.get(
    "/{skill_id}/versions/{version}",
    response_model=SkillVersionResponse,
    description="Retrieve a specific version for a skill.",
    responses={
        200: {
            "description": "Skill version found.",
            "content": {
                "application/json": {
                    "examples": {
                        "version": {
                            "summary": "Single skill version",
                            "value": {
                                "id": "version_01HZX8V5AKMFK7YMRQ7QNXMW9B",
                                "created_at": "2026-04-16T12:15:00Z",
                                "description": "Helps sales reps draft concise outreach.",
                                "directory": "sales_ops_helper",
                                "name": "sales_ops_helper",
                                "skill_id": "skill_01HZX8N83J4WQ2E4K4G3Q2H9E8",
                                "type": "skill_version",
                                "version": "v4",
                            },
                        }
                    }
                }
            },
        }
    },
)
async def get_skill_version(
    request: Request,
    skill_id: str,
    version: str,
    collection: str = Query(
        ...,
        min_length=1,
        max_length=255,
        examples=["acme-prod"],
    ),
    anthropic_beta: Annotated[list[str] | None, Header(alias="anthropic-beta")] = None,
) -> SkillVersionResponse:
    """Get one skill version by version token inside a collection."""
    del anthropic_beta
    skill_version = await _require_skill_version(
        request, skill_id=skill_id, version=version, collection=collection
    )
    return _version_response(skill_version)


@skill_router.delete(
    "/{skill_id}/versions/{version}",
    response_model=SkillVersionDeletedResponse,
    description="Delete a specific skill version.",
    responses={
        200: {
            "description": "Skill version deleted.",
            "content": {
                "application/json": {
                    "examples": {
                        "deleted_version": {
                            "summary": "Deletion marker",
                            "value": {
                                "id": "v4",
                                "type": "skill_version_deleted",
                            },
                        }
                    }
                }
            },
        }
    },
)
async def delete_skill_version(
    request: Request,
    skill_id: str,
    version: str,
    collection: str = Query(
        ...,
        min_length=1,
        max_length=255,
        examples=["acme-prod"],
    ),
    anthropic_beta: Annotated[list[str] | None, Header(alias="anthropic-beta")] = None,
) -> SkillVersionDeletedResponse:
    """Delete a skill version and return a deletion marker payload."""
    del anthropic_beta
    service: SkillService = request.state.injector.get(SkillService)
    skill = await service.get_skill(skill_id=skill_id, collection=collection)
    if not skill:
        raise HTTPException(
            status_code=404,
            detail=f"Skill with id {skill_id} not found in collection {collection}",
        )

    await service.delete_version(
        skill_id=skill_id, version=version, collection=collection
    )
    return SkillVersionDeletedResponse(id=version)


async def _require_skill_version(
    request: Request,
    *,
    skill_id: str,
    version: str,
    collection: str,
) -> SkillVersionEntity:
    service: SkillService = request.state.injector.get(SkillService)
    skill = await service.get_skill(skill_id=skill_id, collection=collection)
    if not skill:
        raise HTTPException(
            status_code=404,
            detail=f"Skill with id {skill_id} not found in collection {collection}",
        )
    skill_version = await service.get_version(
        skill_id=skill_id, version=version, collection=collection
    )
    if not skill_version:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Version {version} for skill {skill_id} "
                f"not found in collection {collection}"
            ),
        )
    return skill_version


def _skill_response(skill: SkillEntity) -> SkillResponse:
    return SkillResponse(
        id=skill.id,
        created_at=skill.created_at,
        display_title=skill.display_title,
        latest_version=skill.latest_version,
        source=skill.source,
        updated_at=skill.updated_at,
        collection=skill.collection,
        loading=skill.loading,
        readonly=skill.readonly,
    )


def _version_response(version: SkillVersionEntity) -> SkillVersionResponse:
    return SkillVersionResponse(
        id=version.id,
        created_at=version.created_at,
        description=version.frontmatter.description,
        directory=version.frontmatter.name,
        name=version.frontmatter.name,
        skill_id=version.skill_id,
        version=version.version,
    )
