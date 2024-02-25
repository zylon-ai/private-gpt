from fastapi import Request, Depends, HTTPException
import logging
from private_gpt.users.core.config import settings
from private_gpt.users.constants.role import Role
from typing import Union, Any, Generator
from datetime import datetime
from private_gpt.users import crud, models, schemas
from private_gpt.users.db.session import SessionLocal
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, SecurityScopes
from private_gpt.users.core.security import (
    ALGORITHM,
    JWT_SECRET_KEY
)
from fastapi import Depends, HTTPException, Security, status
from jose import jwt
from private_gpt.users.utils.audit import log_audit_entry
from pydantic import ValidationError
from private_gpt.users.constants.role import Role
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

def get_current_user(
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


def check_user_role(current_user: models.User = Depends(get_current_user), role: str = ""):
    if current_user.role != role:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have the necessary permissions to perform this action",
        )
    return current_user


def is_company_admin(current_user: models.User = Depends(get_current_user)):
    return check_user_role(current_user, role=Role.ADMIN["name"])


def is_super_admin(current_user: models.User = Depends(get_current_user)):
    return check_user_role(current_user, role=Role.SUPER_ADMIN["name"])
    

def get_company_name(company_id: int, db: Session = Depends(get_db)) -> str:
    company = crud.company.get(db=db, id=company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return company.name


def get_active_subscription(
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    company_id = current_user.user_role.company_id
    if company_id:
        company = crud.company.get(db, company_id)
        if company and company.subscriptions:
            active_subscription = next((sub for sub in company.subscriptions if sub.is_active), None)
            if active_subscription:
                return company

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Access Forbidden - No Active Subscription",
    )


def get_audit_logger(request: Request, db: Session = Depends(get_db)):
    try:
        return lambda model, action, details, user_id=None: log_audit_entry(db, model, action, details, user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in get_audit_logger: {str(e)}")


def get_current_active_user(
    current_user: models.User = Security(get_current_user, scopes=[],),
) -> models.User:
    if not crud.user.is_active(current_user):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user
