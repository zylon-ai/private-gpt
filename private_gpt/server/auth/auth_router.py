"""Authentication and admin REST API endpoints.

Mounted at /v1 when user_auth.enabled is True.

Public endpoints:
    POST /v1/auth/login
    POST /v1/auth/logout  (no-op — token is stateless)
    GET  /v1/auth/me

Admin-only endpoints (require is_admin in the verified token):
    GET/POST/DELETE /v1/admin/users{/username}
    POST/DELETE     /v1/admin/users/{username}/groups/{group_name}
    GET/POST/DELETE /v1/admin/groups{/group_name}
    POST/DELETE     /v1/admin/groups/{group_name}/collections/{collection_name}
    GET/POST/DELETE /v1/admin/collections{/collection_name}
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from private_gpt.server.auth.token_service import TokenService
from private_gpt.server.auth.user_store import CollectionRecord, GroupRecord, UserRecord, UserStore

logger = logging.getLogger(__name__)

auth_router = APIRouter(prefix="/v1")

# ── Pydantic schemas ─────────────────────────────────────────────────────────


class LoginBody(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str
    is_admin: bool
    collections: list[str]


class MeResponse(BaseModel):
    username: str
    is_admin: bool
    collections: list[str]


class CreateUserBody(BaseModel):
    username: str
    password: str
    is_admin: bool = False


class CreateGroupBody(BaseModel):
    group_name: str


class AssignCollectionBody(BaseModel):
    collection_name: str


class CreateCollectionBody(BaseModel):
    collection_name: str
    display_name: str = ""


# ── Auth dependencies ────────────────────────────────────────────────────────


def _get_token_service(request: Request) -> TokenService:
    return request.state.injector.get(TokenService)


def _get_user_store(request: Request) -> UserStore:
    return request.state.injector.get(UserStore)


def _extract_token(authorization: Annotated[str, Header()] = "") -> str:
    if authorization.startswith("Bearer "):
        return authorization[7:]
    return authorization


def _verified_username(
    request: Request,
    authorization: Annotated[str, Header()] = "",
) -> str:
    token_svc: TokenService = request.state.injector.get(TokenService)
    raw = authorization[7:] if authorization.startswith("Bearer ") else authorization
    username = token_svc.verify_token(raw)
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return username


def _verified_admin(
    username: Annotated[str, Depends(_verified_username)],
    request: Request,
) -> str:
    store: UserStore = request.state.injector.get(UserStore)
    if not store.is_admin(username):
        raise HTTPException(status_code=403, detail="Admin access required")
    return username


# ── Public endpoints ─────────────────────────────────────────────────────────


@auth_router.post("/auth/login", tags=["Auth"])
def login(body: LoginBody, request: Request) -> LoginResponse:
    """Authenticate and receive a signed session token."""
    store: UserStore = _get_user_store(request)
    token_svc: TokenService = _get_token_service(request)

    if not store.verify_password(body.username, body.password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    user = store.get_user(body.username)
    token = token_svc.create_token(body.username)
    collections = store.get_user_collections(body.username)

    return LoginResponse(
        token=token,
        username=body.username,
        is_admin=user.is_admin if user else False,
        collections=collections,
    )


@auth_router.post("/auth/logout", tags=["Auth"], status_code=204)
def logout() -> None:
    """Invalidate the current session (client should discard the token)."""
    # Tokens are stateless — nothing to do server-side.


@auth_router.get("/auth/me", tags=["Auth"])
def me(
    username: Annotated[str, Depends(_verified_username)],
    request: Request,
) -> MeResponse:
    """Return information about the currently authenticated user."""
    store: UserStore = _get_user_store(request)
    user = store.get_user(username)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return MeResponse(
        username=username,
        is_admin=user.is_admin,
        collections=store.get_user_collections(username),
    )


# ── Admin: Users ─────────────────────────────────────────────────────────────


@auth_router.get("/admin/users", tags=["Admin"])
def list_users(
    _: Annotated[str, Depends(_verified_admin)],
    request: Request,
) -> list[UserRecord]:
    return _get_user_store(request).list_users()


@auth_router.post("/admin/users", tags=["Admin"])
def create_user(
    body: CreateUserBody,
    _: Annotated[str, Depends(_verified_admin)],
    request: Request,
) -> UserRecord:
    store = _get_user_store(request)
    if store.get_user(body.username) is not None:
        raise HTTPException(status_code=409, detail="User already exists")
    return store.create_user(body.username, body.password, is_admin=body.is_admin)


@auth_router.delete("/admin/users/{username}", tags=["Admin"], status_code=204)
def delete_user(
    username: str,
    admin_username: Annotated[str, Depends(_verified_admin)],
    request: Request,
) -> None:
    if username == admin_username:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    _get_user_store(request).delete_user(username)


@auth_router.post("/admin/users/{username}/groups/{group_name}", tags=["Admin"], status_code=204)
def assign_user_to_group(
    username: str,
    group_name: str,
    _: Annotated[str, Depends(_verified_admin)],
    request: Request,
) -> None:
    _get_user_store(request).assign_user_to_group(username, group_name)


@auth_router.delete("/admin/users/{username}/groups/{group_name}", tags=["Admin"], status_code=204)
def remove_user_from_group(
    username: str,
    group_name: str,
    _: Annotated[str, Depends(_verified_admin)],
    request: Request,
) -> None:
    _get_user_store(request).remove_user_from_group(username, group_name)


# ── Admin: Groups ─────────────────────────────────────────────────────────────


@auth_router.get("/admin/groups", tags=["Admin"])
def list_groups(
    _: Annotated[str, Depends(_verified_admin)],
    request: Request,
) -> list[GroupRecord]:
    return _get_user_store(request).list_groups()


@auth_router.post("/admin/groups", tags=["Admin"])
def create_group(
    body: CreateGroupBody,
    _: Annotated[str, Depends(_verified_admin)],
    request: Request,
) -> GroupRecord:
    return _get_user_store(request).create_group(body.group_name)


@auth_router.delete("/admin/groups/{group_name}", tags=["Admin"], status_code=204)
def delete_group(
    group_name: str,
    _: Annotated[str, Depends(_verified_admin)],
    request: Request,
) -> None:
    _get_user_store(request).delete_group(group_name)


@auth_router.post(
    "/admin/groups/{group_name}/collections/{collection_name}",
    tags=["Admin"],
    status_code=204,
)
def assign_collection_to_group(
    group_name: str,
    collection_name: str,
    _: Annotated[str, Depends(_verified_admin)],
    request: Request,
) -> None:
    _get_user_store(request).assign_collection_to_group(group_name, collection_name)


@auth_router.delete(
    "/admin/groups/{group_name}/collections/{collection_name}",
    tags=["Admin"],
    status_code=204,
)
def remove_collection_from_group(
    group_name: str,
    collection_name: str,
    _: Annotated[str, Depends(_verified_admin)],
    request: Request,
) -> None:
    _get_user_store(request).remove_collection_from_group(group_name, collection_name)


# ── Admin: Collections ────────────────────────────────────────────────────────


@auth_router.get("/admin/collections", tags=["Admin"])
def list_collections(
    _: Annotated[str, Depends(_verified_admin)],
    request: Request,
) -> list[CollectionRecord]:
    return _get_user_store(request).list_collections()


@auth_router.post("/admin/collections", tags=["Admin"])
def create_collection(
    body: CreateCollectionBody,
    _: Annotated[str, Depends(_verified_admin)],
    request: Request,
) -> CollectionRecord:
    return _get_user_store(request).create_collection(
        body.collection_name, body.display_name
    )


@auth_router.delete("/admin/collections/{collection_name}", tags=["Admin"], status_code=204)
def delete_collection(
    collection_name: str,
    _: Annotated[str, Depends(_verified_admin)],
    request: Request,
) -> None:
    _get_user_store(request).delete_collection(collection_name)
