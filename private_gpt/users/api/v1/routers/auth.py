import traceback
from typing import Any, Optional
from datetime import timedelta, datetime

from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi import APIRouter, Body, Depends, HTTPException, Security, status

from private_gpt.users.api import deps
from private_gpt.users.core import security
from private_gpt.users.constants.role import Role
from private_gpt.users.core.config import settings
from private_gpt.users import crud, models, schemas
from private_gpt.users.utils import send_registration_email, Ldap
import logging

logger = logging.getLogger(__name__)

LDAP_SERVER = settings.LDAP_SERVER
# LDAP_ENABLE = settings.LDAP_ENABLE
LDAP_ENABLE = False

router = APIRouter(prefix="/auth", tags=["auth"])

def register_user(
    db: Session,
    email: str,
    fullname: str,
    password: str,
    company: Optional[models.Company] = None,
    department: Optional[models.Department] = None,
) -> models.User:
    """
    Register a new user in the database.
    """
    logging.info(f"User : {email} Password: {password} company_id: {company.id} deparment_id: {department.id}")
    user_in = schemas.UserCreate(
            email=email,
            password=password,
            fullname=fullname,
            company_id=company.id,
            department_id=department.id,
        )    
    try:
        send_registration_email(fullname, email, password)
    except Exception as e:
        logging.info(f"Failed to send registration email: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Failed to send registration email.")
    return crud.user.create(db, obj_in=user_in)


def ldap_login(db, username, password):
    ldap = Ldap(LDAP_SERVER, username, password)
    print("LDAP LOGIN:: ", ldap.who_am_i())
    if not ldap:
        raise HTTPException(
            status_code=400, detail="Incorrect email or password"
        )
    return ldap.who_am_i()

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
        "department_id": user.department_id
    }

def ad_user_register(
    db: Session,
    email: str,
    fullname: str,
    password: str,
) -> models.User:
    """
    Register a new user in the database. Company id is directly given here.
    """
    user_in = schemas.UserCreate(email=email, password=password, fullname=fullname, company_id=1)
    print("user is: ", user_in)
    user = crud.user.create(db, obj_in=user_in)
    user_role_name = Role.GUEST["name"]
    company = crud.company.get(db, 1)

    user_role = create_user_role(db, user, user_role_name, company)
    return user


@router.post("/login", response_model=schemas.TokenSchema)
def login_access_token(
    log_audit: models.Audit = Depends(deps.get_audit_logger),
    db: Session = Depends(deps.get_db),
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    if LDAP_ENABLE:
        existing_user = crud.user.get_by_email(db, email=form_data.username)
        
        if existing_user:
            if existing_user.user_role.role.name == "SUPER_ADMIN":
                pass
            else:
                ldap = ldap_login(db=db, username=form_data.username, password=form_data.password)
        else:
            ldap = ldap_login(db=db, username=form_data.username, password=form_data.password)
            ad_user_register(db=db, email=form_data.username,fullname=ldap, password=form_data.password)

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
        "department_id": str(user.department_id),
    }

    response_dict = {
        "access_token": security.create_access_token(
            token_payload, expires_delta=access_token_expires
        ),
        "refresh_token": security.create_refresh_token(
            token_payload, expires_delta=refresh_token_expires
        ),
        "user": token_payload,
        "token_type": "bearer",
    }
    log_audit(model='User', action='login',
              details=token_payload, user_id=user.id)
    return JSONResponse(content=response_dict)


@router.post("/login/refresh-token", response_model=schemas.TokenSchema)
def refresh_access_token(
    db: Session = Depends(deps.get_db),
    refresh_token: str = Body(..., embed=True),
) -> Any:
    token_payload = security.verify_refresh_token(refresh_token)

    if not token_payload:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    access_token_expires = timedelta(
        minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(
        minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)

    response_dict = {
        "access_token": security.create_access_token(token_payload, expires_delta=access_token_expires),
        "refresh_token": security.create_refresh_token(token_payload, expires_delta=refresh_token_expires),
        "token_type": "bearer",
    }
    return JSONResponse(content=response_dict)


@router.post("/register", response_model=schemas.TokenSchema)
def register(
    *,
    log_audit: models.Audit = Depends(deps.get_audit_logger),

    db: Session = Depends(deps.get_db),
    email: str = Body(...),
    fullname: str = Body(...),
    # password: str = Body(...),
    company_id: int = Body(None, title="Company ID",
                           description="Company ID for the user (if applicable)"),
    department_id: str = Body(None, title="Department ID",
                                description="Department name for the user (if applicable)"),
    role_name: str = Body(None, title="Role Name",
                          description="User role name (if applicable)"),
    current_user: models.User = Security(
        deps.get_current_active_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"]],
    ),
) -> Any:
    """
    Register new user with optional company assignment and role selection.
    """

    existing_user = crud.user.get_by_email(db, email=email)
    if existing_user:
        log_audit(model='User', action='creation',
                  details={"status": '409', 'detail': "The user with this email already exists!", }, user_id=current_user.id)
        raise HTTPException(
            status_code=409,
            detail="The user with this email already exists!",
        )
    random_password = security.generate_random_password()
    try:
        if company_id:
            company = crud.company.get(db, company_id)
            if not company:
                raise HTTPException(
                    status_code=404,
                    detail="Company not found.",
                )
            if department_id:
                department = crud.department.get_by_id(
                    db=db, id=department_id)
                if not department:
                    raise HTTPException(
                        status_code=404,
                        detail="Department not found.",
                    )
                logging.info(f"Department is {department}")
            user = register_user(
                db, email, fullname, random_password, company, department
            )
            user_role_name = role_name or Role.GUEST["name"]
            user_role = create_user_role(db, user, user_role_name, company)
            log_audit(model='user_roles', action='creation',
                      details={"status": '201', 'detail': "User role created successfully.", }, user_id=current_user.id)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Unable to create account.",
        )
    token_payload = create_token_payload(user, user_role)
    response_dict = {
        "access_token": security.create_access_token(token_payload, expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)),
        "refresh_token": security.create_refresh_token(token_payload, expires_delta=timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)),
        "token_type": "bearer",
        "password": random_password,
    }
    log_audit(model='User', action='creation',
              details={"status": '201', 'detail': "User created successfully.", }, user_id=current_user.id)

    return JSONResponse(content=response_dict, status_code=status.HTTP_201_CREATED)
