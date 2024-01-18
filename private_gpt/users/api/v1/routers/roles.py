from typing import Any, List

from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi import APIRouter, Depends, status, Security

from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users import crud, schemas, models


router = APIRouter(prefix='/roles', tags=['roles'])


@router.get("/", response_model=List[schemas.Role])
def get_roles(
    db: Session = Depends(deps.get_db), skip: int = 0, limit: int = 100,
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> Any:
    """
    Retrieve all available user roles.
    """
    roles = crud.role.get_multi(db, skip=skip, limit=limit)
    
    roles_data = [{"id": role.id, "name": role.name} for role in roles]
    
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"message": "Roles retrieved successfully", "roles": roles_data},
    )