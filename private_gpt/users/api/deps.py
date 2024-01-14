import logging
from private_gpt.users.core.config import settings
from private_gpt.users.constants.role import Role
from typing import Union, Any, Generator
from datetime import datetime
from private_gpt.users import crud, models, schemas
from private_gpt.users.db.session import SessionLocal
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from private_gpt.users.core.security import (
    ALGORITHM,
    JWT_SECRET_KEY
)
from fastapi import Depends, HTTPException, Security, status
from jose import jwt
from pydantic import ValidationError
from private_gpt.users.schemas.token import TokenPayload
from sqlalchemy.orm import Session

reusable_oauth2 = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    scopes={
        Role.GUEST["name"]: Role.GUEST["description"],
        # Role.ACCOUNT_ADMIN["name"]: Role.ACCOUNT_ADMIN["description"],
        # Role.ACCOUNT_MANAGER["name"]: Role.ACCOUNT_MANAGER["description"],
        Role.ADMIN["name"]: Role.ADMIN["description"],
        Role.SUPER_ADMIN["name"]: Role.SUPER_ADMIN["description"],
    },
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_db() -> Generator:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

async def get_current_user(
        security_scopes: SecurityScopes,
        db: Session = Depends(get_db), 
        token: str = Depends(reusable_oauth2)
    ) -> models.User:
    if security_scopes.scopes:
        authenticate_value = f'Bearer scope="{security_scopes.scope_str}"'
    else:
        authenticate_value = "Bearer"

    credentials_exception = HTTPException(
        status_code=401,
        detail="os",
        headers={"WWW-Authenticate": authenticate_value},
    )
    try:
        payload = jwt.decode(
            token, JWT_SECRET_KEY, algorithms=[ALGORITHM]
        )
        if payload.get("id") is None:
            raise credentials_exception
        print(payload)
        token_data = schemas.TokenPayload(**payload)
    except (jwt.JWTError, ValidationError):
        logger.error("Error Decoding Token", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
        )
    user = crud.user.get(db, id=token_data.id)
    if not user:
        raise credentials_exception
    if security_scopes.scopes and not token_data.role:
        raise HTTPException(
            status_code=401,
            detail="Not enough permissions",
            headers={"WWW-Authenticate": authenticate_value},
        )
    if (
        security_scopes.scopes
        and token_data.role not in security_scopes.scopes
    ):
        raise HTTPException(
            status_code=401,
            detail="Not enough permissions",
            headers={"WWW-Authenticate": authenticate_value},
        )
    return user


async def get_current_active_user(
    current_user: models.User = Security(get_current_user, scopes=[],),
) -> models.User:
    if not crud.user.is_active(current_user):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user