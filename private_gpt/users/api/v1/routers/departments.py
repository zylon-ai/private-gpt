from typing import Any, List

from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Depends, HTTPException, status, Security

from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users import crud, models, schemas


router = APIRouter(prefix="/departments", tags=["Deparments"])


@router.get("", response_model=List[schemas.Department])
def list_deparments(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> List[schemas.Department]:
    """
    Retrieve a list of companies with pagination support.
    """
    deparments = crud.deparment.get_multi(db, skip=skip, limit=limit)
    return deparments


@router.post("/create", response_model=schemas.Department)
def create_deparment(
    company_in: schemas.DepartmentCreate,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> schemas.Company:
    """
    Create a new company
    """
    deparment = crud.deparment.create(db=db, obj_in=company_in)
    deparment = jsonable_encoder(deparment)

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "message": "Department created successfully",
            "department": deparment
        },
    )


@router.get("/{deparment_id}", response_model=schemas.Department)
def read_company(
    deparment_id: int,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> schemas.Company:
    """
    Read a company by ID
    """
    deparment = crud.deparment.get_by_id(db, id=deparment_id)
    if deparment is None:
        raise HTTPException(status_code=404, detail="Deparment not found")
    return deparment


@router.put("/{deparment_id}", response_model=schemas.Department)
def update_company(
    deparment_id: int,
    deparment_in: schemas.DepartmentUpdate,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> schemas.Company:
    """
    Update a company by ID
    """
    deparment = crud.deparment.get_by_id(db, id=deparment_id)
    if deparment is None:
        raise HTTPException(status_code=404, detail="Deparment not found")

    updated_deparment = crud.deparment.update(
        db=db, db_obj=deparment, obj_in=deparment_in)
    updated_deparment = jsonable_encoder(updated_deparment)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": f"{deparment_in} Deparment updated successfully",
            "deparment": updated_deparment
        },
    )


@router.delete("/{deparment_id}", response_model=schemas.Department)
def delete_company(
    deparment_id: int,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> schemas.Company:
    """
    Delete a company by ID
    """

    deparment = crud.deparment.remove(db=db, id=deparment_id)
    if deparment is None:
        raise HTTPException(status_code=404, detail="Deparment not found")
    deparment = jsonable_encoder(deparment)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": "Deparment deleted successfully",
            "deparment": deparment
        },
    )
