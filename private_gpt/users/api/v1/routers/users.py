from typing import Any, List
from private_gpt.users import crud, models, schemas
from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users.core.config import settings
from private_gpt.users.core.security import verify_password, get_password_hash
from fastapi import APIRouter, Body, Depends, HTTPException, Security
from fastapi.encoders import jsonable_encoder
from pydantic.networks import EmailStr
from sqlalchemy.orm import Session

router = APIRouter(prefix="/users", tags=["users"])


@router.get("", response_model=List[schemas.User])
def read_users(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"]],
    ),
) -> Any:
    """
    Retrieve all users.
    """
    users = crud.user.get_multi(db, skip=skip, limit=limit,)
    return users


@router.post("", response_model=schemas.User)
def create_user(
    *,
    db: Session = Depends(deps.get_db),
    user_in: schemas.UserCreate,
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"]],
    ),
) -> Any:
    """
    Create new user.
    """
    user = crud.user.get_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=409,
            detail="The user with this username already exists in the system.",
        )
    user = crud.user.create(db, obj_in=user_in)
    return user


@router.put("/me", response_model=schemas.User)
def update_user_me(
    *,
    db: Session = Depends(deps.get_db),
    fullname: str = Body(None),
    email: EmailStr = Body(None),
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Update own user.
    """
    current_user_data = jsonable_encoder(current_user)
    print("Current user data: ", current_user_data)
    user_in = schemas.UserUpdate(**current_user_data)
    if fullname is not None:
        user_in.fullname = fullname
    if email is not None:
        user_in.email = email
    print(f"DB obj: {current_user}\n obj IN : {user_in}")
    user = crud.user.update(db, db_obj=current_user, obj_in=user_in)
    return user


@router.get("/me", response_model=schemas.User)
def read_user_me(
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Get current user.
    """
    if not current_user.user_role:
        role = None
    else:
        role = current_user.user_role.role.name
    user_data = schemas.User(
        id=current_user.id,
        email=current_user.email,
        is_active=current_user.is_active,
        fullname=current_user.fullname,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at,
        last_login = current_user.last_login,
        role=role
    )
    return user_data


@router.patch("/me/change-password", response_model=schemas.User)
def change_password(
    *,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
    old_password: str = Body(..., embed=True),
    new_password: str = Body(..., embed=True),
) -> Any:
    """
    Change current user's password.
    """
    # Verify the old password
    if not verify_password(old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Old password is incorrect")

    # Change the password
    new_password_hashed = get_password_hash(new_password)
    current_user.hashed_password = new_password_hashed
    db.commit()

    role = current_user.user_role.role.name if current_user.user_role else None
    user_data = schemas.User(
        id=current_user.id,
        email=current_user.email,
        is_active=current_user.is_active,
        fullname=current_user.fullname,
        created_at=current_user.created_at,
        updated_at=current_user.updated_at,
        last_login=current_user.last_login,
        role=role,
    )
    return user_data


@router.get("/{user_id}", response_model=schemas.User)
def read_user_by_id(
    user_id: int,
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"]],
    ),
    db: Session = Depends(deps.get_db),
) -> Any:
    """
    Get a specific user by id.
    """
    if user_id is None:
        return "User id is not given."
    user = crud.user.get(db, id=user_id)
    return user


@router.put("/{user_id}", response_model=schemas.User)
def update_user(
    *,
    db: Session = Depends(deps.get_db),
    user_id: int,
    user_in: schemas.UserUpdate,
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"]],
    ),
) -> Any:
    """
    Update a user.
    """
    user = crud.user.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=404,
            detail="The user with this username does not exist in the system",
        )
    user = crud.user.update(db, db_obj=user, obj_in=user_in)
    return user