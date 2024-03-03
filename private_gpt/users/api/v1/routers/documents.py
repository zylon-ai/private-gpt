import traceback
import logging
from typing import Any, List
from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Depends, HTTPException, status, Security, Request

from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users import crud, models, schemas
from private_gpt.users.schemas import Document

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/documents', tags=['Documents'])


@router.get("", response_model=List[schemas.Document])
def list_files(
    request: Request,
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"], Role.ADMIN["name"]],
    )
):
    def get_department_name(db, id):
        dep = crud.department.get_by_id(db=db, id=id)
        return dep.name

    def get_username(db, id):
        user = crud.user.get_by_id(db=db, id=id) 
        return user.fullname
    try:
        role = current_user.user_role.role.name if current_user.user_role else None
        if role == "SUPER_ADMIN":
            docs = crud.documents.get_multi(db, skip=skip, limit=limit)
        else:
            docs = crud.documents.get_multi_documents(
                db, department_id=current_user.department_id, skip=skip, limit=limit)
            
        docs = [
            schemas.Document(
                id=doc.id,
                filename=doc.filename,
                uploaded_at=doc.uploaded_at,
                uploaded_by=get_username(db, doc.uploaded_by),
                department=get_department_name(db, doc.department_id)
            )
            for doc in docs
        ]
        return docs
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"There was an error listing the file(s).")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        )



@router.get('{department_id}', response_model=List[schemas.Document])
def list_files_by_department(
    request: Request,
    department_id: int,
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    )
):
    '''
    Listing the documents by the department id
    '''
    try:
        docs = crud.documents.get_multi_documents(
            db, department_id=department_id, skip=skip, limit=limit)
        return docs
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"There was an error listing the file(s).")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error.",
        )


@router.get('/files', response_model=List[schemas.DocumentList])
def list_files_by_department(
    request: Request,
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.ADMIN["name"]],
    )
):
    '''
    Listing the documents by the ADMIN of the Department
    '''
    try:
        department_id = current_user.department_id
        docs = crud.documents.get_multi_documents(
            db, department_id=department_id, skip=skip, limit=limit)
        return docs
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"There was an error listing the file(s).")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error.",
        )