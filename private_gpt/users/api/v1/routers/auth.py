from typing import Any, Optional
from datetime import timedelta, datetime

from sqlalchemy.orm import Session
from pydantic.networks import EmailStr
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import APIRouter, Body, Depends, HTTPException, Security, Path, status

from private_gpt.users.api import deps
from private_gpt.users.core import security
from private_gpt.users.constants.role import Role
from private_gpt.users.core.config import settings
from private_gpt.users import crud, models, schemas
from private_gpt.users.utils import send_registration_email


router = APIRouter(prefix="/auth", tags=["auth"])

def register_user(
    db: Session,
    email: str,
    fullname: str,
    password: str,
    company: Optional[models.Company] = None,
) -> models.User:
    """
    Register a new user in the database.
    """
    print(f"{email} {fullname} {password} {company.id}")
    user_in = schemas.UserCreate(email=email, password=password, fullname=fullname, company_id=company.id)
    send_registration_email(fullname, email, password)
    return crud.user.create(db, obj_in=user_in)


def create_user_role(
    db: Session,
    user: models.User,
    role_name: str,
    company: Optional[models.Company] = None,
) -> models.UserRole:
    """
    Create a user role in the database.
    """
    role = crud.role.get_by_name(db, name=role_name)
    user_role_in = schemas.UserRoleCreate(user_id=user.id, role_id=role.id, company_id=company.id if company else None)
    return crud.user_role.create(db, obj_in=user_role_in)


def create_token_payload(user: models.User, user_role: models.UserRole) -> dict:
    """
    Create a token payload for authentication.
    """
    return {
        "id": str(user.id),
        "email": str(user.email),
        "role": user_role.role.name,
        "username": str(user.fullname),
        "company_id": user_role.company.id if user_role.company else None,
    }


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

    access_token_expires = timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    refresh_token_expires = timedelta(
        minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES
    )
    user_in = schemas.UserUpdate(
        email=user.email,
        fullname=user.fullname,
        company_id=user.company_id,
        last_login=datetime.now()
    )

    user = crud.user.update(db, db_obj=user, obj_in=user_in)

    if user.user_role:
        role = user.user_role.role.name
        if user.user_role.company_id:
            company_id = user.user_role.company_id
        else: company_id = None

    token_payload = {
        "id": str(user.id),
        "email": str(user.email),
        "username": str(user.fullname),

        "role": role,
        "company_id": company_id,
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

@router.post("/login/refresh-token", response_model=schemas.TokenSchema)
def refresh_access_token(
    db: Session = Depends(deps.get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Any:
    """
    Refresh access token using a valid refresh token
    """
    refresh_token = form_data.refresh_token
    token_payload = security.verify_refresh_token(refresh_token)

    if not token_payload:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)

    return {
        "access_token": security.create_access_token(
            token_payload, expires_delta=access_token_expires
        ),
        "refresh_token": security.create_refresh_token(
            token_payload, expires_delta=refresh_token_expires
        ),
        "token_type": "bearer",
    }

@router.post("/{company_name}/register", response_model=schemas.User)
def register_for_company(
    *,
    db: Session = Depends(deps.get_db),
    email: str = Body(...),
    fullname: str = Body(...),
    company_name: Optional[str] = Path(..., title="Company Name", description="Only for company admin"),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"], Role.ADMIN['name']],
    ),
) -> Any:
    """
    Register new user for a specific company.
    """
    user = crud.user.get_by_email(db, email=email)
    if user:
        raise HTTPException(
            status_code=409,
            detail="The user with this username already exists in the system",
        )

    if current_user.user_role.role.name not in {Role.ADMIN["name"], Role.SUPER_ADMIN["name"]}:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to register users for a company.",
        )

    company = crud.company.get_by_company_name(db, company_name=company_name)
    print(f"Company is : {company.id}")
    if not (current_user.user_role.role.name == Role.ADMIN["name"] and current_user.user_role.company_id == company.id):
        raise HTTPException(
            status_code=403,
            detail="You are not the admin of the specified company.",
        )
    
    random_password = security.generate_random_password()
    user = register_user(db, email, fullname, random_password, company)
    user_role = create_user_role(db, user, Role.GUEST["name"], company)

    token_payload = create_token_payload(user, user_role)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"message": "User registered successfully.\n\n Check respective user email for login credentials", "user": jsonable_encoder(user)},
    )


@router.post("/register", response_model=schemas.TokenSchema)
def register_without_company_assignment(
    *,
    db: Session = Depends(deps.get_db),
    email: str = Body(...),
    fullname: str = Body(...),
    company_id: int = Body(None, title="Company ID", description="Company ID for the user (if applicable)"),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"], Role.ADMIN['name']],
    ),
) -> Any:
    """
    Register new user with company assignment.
    """
    user = crud.user.get_by_email(db, email=email)
    if user:
        raise HTTPException(
            status_code=409,
            detail="The user with this username already exists in the system",
        )

    if current_user.user_role.role.name != Role.SUPER_ADMIN["name"]:
        raise HTTPException(
            status_code=403,
            detail="You do not have permission to register users without a company.",
        )

    if company_id is None:
        raise HTTPException(
            status_code=400,
            detail="Company ID is required for registering a user without a specific company.",
        )
    
    random_password = security.generate_random_password()
    company = crud.company.get(db, company_id)
    user = register_user(db, email, fullname, random_password, company)
    user_role = create_user_role(db, user, Role.ADMIN["name"], company)

    token_payload = create_token_payload(user, user_role)
    return {
        "access_token": security.create_access_token(token_payload, expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)),
        "refresh_token": security.create_refresh_token(token_payload, expires_delta=timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)),
        "token_type": "bearer",
    }

