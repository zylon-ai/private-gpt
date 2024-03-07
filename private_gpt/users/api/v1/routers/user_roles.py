from typing import Any

from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Depends, HTTPException, Security, status

from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users import crud, models, schemas

router = APIRouter(prefix="/user-roles", tags=["user-roles"])

@router.post("", response_model=schemas.UserRole)
def assign_user_role(
    *,
    db: Session = Depends(deps.get_db),
    user_role_in: schemas.UserRoleCreate,
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[
            Role.SUPER_ADMIN["name"],
            Role.OPERATOR["name"],  
        ],
    ),
) -> Any:
    """
    Assign a role to a user after creation of a user
    """
    user_role = crud.user_role.get_by_user_id(db, user_id=user_role_in.user_id)
    if user_role:
        raise HTTPException(
            status_code=409,
            detail="This user has already been assigned a role.",
        )
    user_role = crud.user_role.create(db, obj_in=user_role_in)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={"message": "User role assigned successfully", "user_role": jsonable_encoder(user_role)},
    )


@router.put("/{user_id}", response_model=schemas.UserRole)
def update_user_role(
    *,
    db: Session = Depends(deps.get_db),
    user_id: int,
    user_role_in: schemas.UserRoleUpdate,
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[
            Role.SUPER_ADMIN["name"],
            Role.OPERATOR["name"],  
        ],
    ),
) -> Any:
    """
    Update a user's role.
    """
    user_role = crud.user_role.get_by_user_id(db, user_id=user_id)
    if not user_role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="There is no role assigned to this user",
        )
    user_role = crud.user_role.update(
        db, db_obj=user_role, obj_in=user_role_in
    )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "User role updated successfully", "user_role": jsonable_encoder(user_role)},
    )


