from typing import Any, List

from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Depends, HTTPException, status, Security

from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users import crud, models, schemas


router = APIRouter(prefix="/departments", tags=["Departments"])


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
    Retrieve a list of department with pagination support.
    """
    deparments = crud.department.get_multi(db, skip=skip, limit=limit)
    return deparments


@router.post("/create", response_model=schemas.Department)
def create_deparment(
    department_in: schemas.DepartmentCreate,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> schemas.Department:
    """
    Create a new department
    """
    company_id = current_user.company_id
    department_create_in = schemas.DepartmentAdminCreate(name=department_in.name, company_id=company_id)
    department = crud.department.create(db=db, obj_in=department_create_in)
    department = jsonable_encoder(department)

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "message": "Department created successfully",
            "department": department
        },
    )


@router.post("/read", response_model=schemas.Department)
def read_department(
    department_id: int,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> schemas.Department:
    """
    Read a Department by ID
    """
    department = crud.department.get_by_id(db, id=department_id)
    if department is None:
        raise HTTPException(status_code=404, detail="department not found")
    return department


@router.post("/update", response_model=schemas.Department)
def update_department(
    department_in: schemas.DepartmentUpdate,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> schemas.Department:
    """
    Update a Department by ID
    """
    department = crud.department.get_by_id(db, id=department_in.id)
    if department is None:
        raise HTTPException(status_code=404, detail="department not found")

    updated_department = crud.department.update(
        db=db, db_obj=department, obj_in=department_in)
    updated_department = jsonable_encoder(updated_department)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": f"{department_in} department updated successfully",
            "department": updated_department
        },
    )


@router.post("/delete", response_model=schemas.Department)
def delete_department(
    department_in: schemas.DepartmentDelete,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> schemas.Department:
    """
    Delete a Department by ID
    """
    department_id = department_in.id
    department = crud.department.get(db, id=department_id)
    if department is None:
        raise HTTPException(status_code=404, detail="User not found")

    department = crud.department.remove(db=db, id=department_id)
    if department is None:
        raise HTTPException(status_code=404, detail="department not found")
    department = jsonable_encoder(department)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": "Department deleted successfully",
            "deparment": department,
        },
    )
