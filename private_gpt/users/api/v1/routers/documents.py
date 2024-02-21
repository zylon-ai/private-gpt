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
        scopes=[Role.SUPER_ADMIN["name"]],
    )
):
    try:
        docs = crud.documents.get_multi(db, skip=skip, limit=limit)
        return docs
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"There was an error listing the file(s).")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error: Unable to ingest file.",
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
            detail="Internal Server Error: Unable to ingest file.",
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
            detail="Internal Server Error: Unable to ingest file.",
        )
