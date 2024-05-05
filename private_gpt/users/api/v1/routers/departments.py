import logging
import traceback

from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi_pagination import Page, paginate

from fastapi import APIRouter, Depends, HTTPException, status, Security, Request

from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users import crud, models, schemas

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/departments", tags=["Departments"])


def log_audit_department(
    request: Request,
    db: Session,
    current_user: models.User,
    action: str,
    details: dict,
):
    try:
        audit_entry = models.Audit(
            user_id=current_user.id,
            model='Department',
            action=action,
            details=details,
            ip_address=request.client.host,
        )
        db.add(audit_entry)
        db.commit()
    except Exception as e:
        print(traceback.format_exc())


@router.get("", response_model=Page[schemas.Department])
def list_departments(
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
    ),
) -> Page[schemas.Department]:
    """
    Retrieve a list of departments with pagination support.
    """
    try:
        role = current_user.user_role.role.name if current_user.user_role else None
        if role == "SUPER_ADMIN":
            deps = crud.department.get_multi(db)
        else:
            deps = crud.department.get_multi_department(
                db, department_id=current_user.department_id)
        deps = [
            schemas.Department(
                id=dep.id,
                name=dep.name,
                total_users=dep.total_users,
                total_documents=dep.total_documents,
            )
            for dep in deps
        ]
        return paginate(deps)
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"There was an error listing the file(s).")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        )


@router.post("/create", response_model=schemas.Department)
def create_department(
    request: Request,
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
    try:
        company_id = current_user.company_id
        department_create_in = schemas.DepartmentAdminCreate(
            name=department_in.name, company_id=company_id)
        department = crud.department.create(db=db, obj_in=department_create_in)
        department1 = jsonable_encoder(department)

        details = {
            'detail': 'Department created successfully',
            'department_id': department.id,
            'department_name': department.name
        }

        log_audit_department(request, db, current_user, 'create', details)

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "message": "Department created successfully",
                "department": department1
            },
        )
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error creating department: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
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
    try:
        department = crud.department.get_by_id(db, id=department_id)
        if department is None:
            raise HTTPException(status_code=404, detail="Department not found")
        return department
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error reading department: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        )


@router.post("/update", response_model=schemas.Department)
def update_department(
    request: Request,
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
    try:
        department = crud.department.get_by_id(db, id=department_in.id)
        old_name = department.name
        if department is None:
            raise HTTPException(status_code=404, detail="Department not found")

        updated_department = crud.department.update(
            db=db, db_obj=department, obj_in=department_in)
        details = {
            'department_id': department.id,
            'before': {
                'name': old_name
            },
            'after': {
                'name': department_in.name
            }
        }
        log_audit_department(request, db, current_user, 'update', details)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": f"Department updated successfully",
            },
        )
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error updating department: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        )


@router.post("/delete", response_model=schemas.Department)
def delete_department(
    request: Request,
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
    try:
        department_id = department_in.id
        department = crud.department.get(db, id=department_id)
        if department is None:
            raise HTTPException(status_code=404, detail="Department not found")

        details = {
            'department_id': department.id,
            'department_name': department.name
        }
        crud.department.remove(db=db, id=department_id)
        log_audit_department(request, db, current_user, 'delete', details)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Department deleted successfully",
            },
        )
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error deleting department: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        )
