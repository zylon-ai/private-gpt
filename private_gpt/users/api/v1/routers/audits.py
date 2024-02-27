from typing import Any, List

from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Depends, HTTPException, status, Security

from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users import crud, models, schemas


router = APIRouter(prefix="/audit", tags=["Companies"])


@router.get("", response_model=List[schemas.Audit])
def list_companies(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> List[schemas.Audit]:
    """
    Retrieve a list of companies with pagination support.
    """
    def get_fullname(id):
        user = crud.user.get_by_id(db, id=id)
        if user:
            return user.fullname
        return ""
    
    logs = crud.audit.get_multi_desc(db, skip=skip, limit=limit)
    logs = [
            schemas.Audit(
                id=dep.id,
                model=dep.model,
                username=get_fullname(dep.user_id),
                details=dep.details,
                action=dep.action,
                timestamp=dep.timestamp,
            )
            for dep in logs
        ]
    return logs
