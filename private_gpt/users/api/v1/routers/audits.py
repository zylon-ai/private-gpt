from typing import Any, List

from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Depends, HTTPException, status, Security, Request

from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users import crud, models, schemas


router = APIRouter(prefix="/audit", tags=["Companies"])


@router.get("", response_model=List[schemas.Audit])
def list_auditlog(
    request: Request,
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> List[schemas.Audit]:
    """
    Retrieve a list of audit logs with pagination support.
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
                ip_address=dep.ip_address,
            )
            for dep in logs
        ]
    return logs



@router.post("", response_model=schemas.Audit)
def get_auditlog(
    request: Request,
    audit: schemas.GetAudit,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
):
    """
    Retrieve a single audit log.
    """
    def get_fullname(id):
        user = crud.user.get_by_id(db, id=id)
        if user:
            return user.fullname
        return ""
    
    logs = crud.audit.get_by_id(db, id=audit.id)
    logs = schemas.Audit(
                id=logs.id,
                model=logs.model,
                username=get_fullname(logs.user_id),
                details=logs.details,
                action=logs.action,
                timestamp=logs.timestamp,
                ip_address=logs.ip_address,
            )
    return logs