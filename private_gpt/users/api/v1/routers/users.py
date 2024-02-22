from typing import Any, List, Optional

from sqlalchemy.orm import Session
from pydantic.networks import EmailStr
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Body, Depends, HTTPException, Security, status, Path

from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users.core.config import settings
from private_gpt.users import crud, models, schemas
from private_gpt.users.core.security import verify_password, get_password_hash

router = APIRouter(prefix="/users", tags=["users"])

@router.get("", response_model=List[schemas.User])
def read_users(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"]],
    ),
) -> Any:
    """                                 
    Retrieve all users.
    """
    role = current_user.user_role.role.name if current_user.user_role else None
    if role == "ADMIN":
        users = crud.user.get_by_department_id(db=db, department_id=current_user.department_id, skip=skip, limit=limit)
    else:
        users = crud.user.get_multi(db, skip=skip, limit=limit)
    return users


@router.get("/company/{company_id}", response_model=List[schemas.User])
def read_users_by_company(
    company_id: int = Path(..., title="Company ID",
                           description="Only for company admin"),
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"]],
    ),
):
    """Retrieve all users of that company only"""
    company = crud.company.get(db, company_id)

    if company is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with ID '{company_id}' not found",
        )

    users = crud.user.get_multi_by_company_id(db, company_id=company.id)
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
            status_code=status.HTTP_409_CONFLICT,
            detail="The user with this email already exists in the system.",
        )
    user = crud.user.create(db, obj_in=user_in)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"message": "User created successfully", "user": jsonable_encoder(user)},
    )


@router.put("/me", response_model=schemas.User)
def update_username(
    *,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
    update_in: schemas.UsernameUpdate,
) -> Any:
    """
    Update own username.
    """
    user_in = schemas.UserUpdate(fullname=update_in.fullname, email=current_user.email, company_id=current_user.company_id)
    user = crud.user.update(db, db_obj=current_user, obj_in=user_in)
    user_data = schemas.UserBaseSchema(
        email=user.email,
        fullname=user.fullname,
        company_id=user.company_id,
        department_id=user.department_id,
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Username updated successfully",
                 "user": jsonable_encoder(user_data)},
    )



@router.get("/me", response_model=schemas.User)
def read_user_me(
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_user),
) -> Any:
    """
    Get current user.
    """
    role = current_user.user_role.role.name if current_user.user_role else None
    user_data = schemas.Profile(
        email=current_user.email,
        fullname=current_user.fullname,
        company_id = current_user.company_id,
        department_id=current_user.department_id,
        role =role
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Current user retrieved successfully", "user": jsonable_encoder(user_data)},
    )


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
    if not verify_password(old_password, current_user.hashed_password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Old password is incorrect")

    new_password_hashed = get_password_hash(new_password)
    current_user.hashed_password = new_password_hashed
    db.commit()

    role = current_user.user_role.role.name if current_user.user_role else None
    user_data = schemas.UserBaseSchema(
        id=current_user.id,
        email=current_user.email,
        fullname=current_user.fullname,
        company_id= current_user.company_id,
        department_id=current_user.department_id,
    )
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Password changed successfully", "user": jsonable_encoder(user_data)},
    )


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
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"message": "User id is not given."})
    user = crud.user.get(db, id=user_id)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "User retrieved successfully", "user": jsonable_encoder(user)},
    )


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
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The user with this id does not exist in the system",
        )
    user = crud.user.update(db, db_obj=user, obj_in=user_in)
    user_data = schemas.UserBaseSchema(
        id=user.id,
        email=user.email,
        fullname=user.fullname,
        company_id=user.company_id,
        department_id=user.department_id,
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "User updated successfully", "user": jsonable_encoder(user_data)},
    )


@router.get("/")
def home_page(
    *,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_active_subscription,
    ),
):
    return JSONResponse(status_code=status.HTTP_200_OK, content={"message": "Welcome to QuickGPT"})



@router.patch("/{user_id}/change-password", response_model=schemas.User)
def admin_change_password(
    *,
    db: Session = Depends(deps.get_db),
    user_id: int,
    new_password: str = Body(..., embed=True),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"]],
    ),
) -> Any:
    """
    Admin/Super Admin change user's password without confirming the previous password.
    """
    user = crud.user.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="The user with this id does not exist in the system",
        )

    new_password_hashed = get_password_hash(new_password)
    user.hashed_password = new_password_hashed
    db.commit()

    user_data = schemas.UserBaseSchema(
        id=user.id,
        email=user.email,
        fullname=user.fullname,
        company_id=user.company_id,
        department_id=user.department_id,
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "User password changed successfully",
                 "user": jsonable_encoder(user_data)},
    )


@router.post("/delete")
def delete_user(
    *,
    db: Session = Depends(deps.get_db),
    delete_user: schemas.DeleteUser,
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"]],
    ),
) -> Any:
    """
    Delete a user by ID.
    """
    user_id = delete_user.id
    user = crud.user.get(db, id=user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    crud.user.remove(db, id=user_id)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "User deleted successfully"},
    )


@router.post("/update_user")
def admin_update_user(
    *,
    db: Session = Depends(deps.get_db),
    user_update: schemas.UserAdminUpdate,
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"], Role.SUPER_ADMIN["name"]],
    ),
) -> Any:
    """
    Update the user by the Admin/Super_ADMIN 
    """
    existing_user = crud.user.get(db, id=user_update.id)

    if existing_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found with id: {user_update.id}",
        )
    if existing_user.fullname == user_update.fullname:
        pass
    else:
        fullname = crud.user.get_by_name(db, name=user_update.fullname)
        if fullname:
            raise HTTPException(
                status_code=409,
                detail="The user with this username already exists!",
            )

    role = crud.role.get_by_name(db, name=user_update.role)
    if role.id == 1:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Cannot create SUPER ADMIN!",
        )

    user_role = crud.user_role.get_by_user_id(db, user_id=existing_user.id)
    role_in = schemas.UserRoleUpdate(
        user_id=existing_user.id,
        role_id=role.id,
    )
    role = crud.user_role.update(db, db_obj=user_role, obj_in=role_in)

    user_in = schemas.UserAdmin(fullname=user_update.fullname, department_id=user_update.department_id)
    crud.user.update(db, db_obj=existing_user, obj_in=user_in)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "User updated successfully",
                 }
    )
