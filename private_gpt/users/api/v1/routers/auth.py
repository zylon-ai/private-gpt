from typing import Any
from datetime import timedelta, datetime

from pydantic.networks import EmailStr
from sqlalchemy.orm import Session
from fastapi import APIRouter, Body, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordRequestForm

from private_gpt.users.api import deps
from private_gpt.users.core import security
from private_gpt.users.constants.role import Role
from private_gpt.users.core.config import settings
from private_gpt.users import crud, models, schemas
from private_gpt.users.utils import send_registration_email


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=schemas.TokenSchema)
def login_access_token(
    db: Session = Depends(deps.get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    user = crud.user.authenticate(
        db, email=form_data.username, password=form_data.password
    )
    if not user:
        raise HTTPException(
            status_code=400, detail="Incorrect email or password"
        )
    # elif not crud.user.is_active(user):
    #     raise HTTPException(status_code=400, detail="Inactive user")
   
    access_token_expires = timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    refresh_token_expires = timedelta(
        minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES
    )
    user_in = schemas.UserUpdate(
        email = user.email,
        fullname = user.fullname,
        last_login=datetime.now()
    )
    user = crud.user.update(db, db_obj=user, obj_in=user_in)
    if not user.user_role:
        role = "GUEST"
    else:
        role = user.user_role.role.name
    
    token_payload = {
        "id": str(user.id),
        "role": role,
    }
    
    return {
        "access_token": security.create_access_token(
            token_payload, expires_delta=access_token_expires
        ),
        "refresh_token": security.create_refresh_token(
            token_payload, expires_delta=refresh_token_expires
        ),
        "token_type": "bearer",
    }


@router.post("/register", response_model=schemas.TokenSchema)
def register(
    *,
    db: Session = Depends(deps.get_db),
    email: EmailStr = Body(...),
    fullname: str = Body(...),
    role: str = Body(Default="GUEST"),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"]],
    ),
) -> Any:
    """
    Register new user.
    """
    user = crud.user.get_by_email(db, email=email)
    if user:
        raise HTTPException(
            status_code=409,
            detail="The user with this username already exists in the system",
        )
    random_password = security.generate_random_password()
    user_in = schemas.UserCreate(
        email=email,
        password=random_password,
        fullname=fullname,
    )
    user = crud.user.create(db, obj_in=user_in)
    send_registration_email(fullname, email, random_password)

    role_db = crud.role.get_by_name(db, name=role)
    if not role_db:
        raise HTTPException(
            status_code=404,
            detail=f"Role '{role}' not found",
        )
    user_role_in = schemas.UserRoleCreate(
        user_id=user.id,
        role_id=role_db.id
    )
    user_role = crud.user_role.create(db, obj_in=user_role_in)
    access_token_expires = timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    refresh_token_expires = timedelta(
        minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES
    )
    token_payload = {
        "id": str(user.id),
        "role": user_role.role.name,  
    }
    return {
        "access_token": security.create_access_token(
            token_payload, expires_delta=access_token_expires
        ),
        "refresh_token": security.create_refresh_token(
            token_payload, expires_delta=refresh_token_expires
        ),
        "token_type": "bearer",
    }
