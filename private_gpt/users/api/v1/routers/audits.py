from typing import Any, List
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, Security
from private_gpt.users import crud, models, schemas
from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users.utils.export import generate_audit_log_report

router = APIRouter(prefix="/audit", tags=["Audit"])

def get_fullname(db: Session, id: int) -> str:
    user = crud.user.get_by_id(db, id=id)
    return user.username if user else ""

def convert_audit_logs(db: Session, logs: List[Any], username: str = None) -> List[schemas.Audit]:
    return [
        schemas.Audit(
            id=dep.id,
            model=dep.model,
            username=get_fullname(db, dep.user_id),
            details=dep.details,
            action=dep.action,
            timestamp=dep.timestamp,
            ip_address=dep.ip_address,
        )
        for dep in logs
    ]

@router.get("", response_model=List[schemas.Audit])
def list_auditlog(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> List[schemas.Audit]:
    logs = crud.audit.get_multi_desc(db, skip=skip, limit=limit)
    return convert_audit_logs(db, logs)

@router.get("filter/", response_model=List[schemas.Audit])
def filter_auditlog(
    db: Session = Depends(deps.get_db),
    filter_in= Depends(schemas.AuditFilter),
    # current_user: models.User = Security(
    #     deps.get_current_user,
    #     scopes=[Role.SUPER_ADMIN["name"]],
    # ),
) -> List[schemas.Audit]:
    logs = crud.audit.filter(db, obj_in=filter_in)
    return convert_audit_logs(db, logs)

@router.get("download/")
def download_auditlog(
    db: Session = Depends(deps.get_db),
    filter_in= Depends(schemas.ExcelFilter),
    # current_user: models.User = Security(
    #     deps.get_current_user,
    #     scopes=[Role.SUPER_ADMIN["name"]],
    # ),
):
    logs = crud.audit.excel_filter(db, obj_in=filter_in)
    username = filter_in.username if filter_in.username else None
    logs = convert_audit_logs(db, logs, username)
    excel_buffer = generate_audit_log_report(logs, username)
    return StreamingResponse(
        iter([excel_buffer.getvalue()]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment;filename=audit_logs.xlsx"},
    )

@router.post("", response_model=schemas.Audit)
def get_auditlog(
    audit: schemas.GetAudit,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
):
    logs = crud.audit.get_by_id(db, id=audit.id)
    return convert_audit_logs(db, [logs])[0]
