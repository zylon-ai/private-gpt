import logging
import traceback

from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Depends, HTTPException, status, Security

from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users import crud, models, schemas

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/departments", tags=["Departments"])


def log_audit_department(
    db: Session,
    current_user: models.User,
    action: str,
    details: dict
):
    try:
        audit_entry = models.Audit(
            user_id=current_user.id,
            model='Department',
            action=action,
            details=details,
        )
        db.add(audit_entry)
        db.commit()
    except Exception as e:
        print(traceback.format_exc())


@router.get("", response_model=list[schemas.Department])
def list_departments(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Security(
        deps.get_current_user,
    ),
) -> list[schemas.Department]:
    """
    Retrieve a list of departments with pagination support.
    """
    try:
        role = current_user.user_role.role.name if current_user.user_role else None
        if role == "SUPER_ADMIN":
            deps = crud.department.get_multi(db, skip=skip, limit=limit)
        else:
            deps = crud.department.get_multi_department(
                db, department_id=current_user.department_id, skip=skip, limit=limit)
            
        deps = [
            schemas.Department(
                id=dep.id,
                name=dep.name,
                total_users=dep.total_users,
                total_documents=dep.total_documents,
            )
            for dep in deps
        ]
        return deps
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"There was an error listing the file(s).")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        )


@router.post("/create", response_model=schemas.Department)
def create_department(
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
            'user_id': current_user.id,
            'department_id': department.id,
            'department_name': department.name
        }

        log_audit_department(db, current_user, 'create', details)

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

        details = {
            'status': 200,
            'user_id': current_user.id,
            'department_id': department.id,
            'department_name': department.name
        }

        log_audit_department(db, current_user, 'read', details)

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
        updated_department = jsonable_encoder(updated_department)

        details = {
            'status': '200',
            'user_id': current_user.id,
            'department_id': department.id,
            'old_department_name': old_name,
            'new_department_name': department.name,
        }

        log_audit_department(db, current_user, 'update', details)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": f"Department updated successfully",
                "department": updated_department
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
            'status': 200,
            'user_id': current_user.id,
            'department_id': department.id,
            'department_name': department.name
        }

        log_audit_department(db, current_user, 'delete', details)

        deleted_department = crud.department.remove(db=db, id=department_id)
        deleted_department = jsonable_encoder(deleted_department)


        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Department deleted successfully",
                "department": deleted_department,
            },
        )
    except Exception as e:
        print(traceback.format_exc())
        logger.error(f"Error deleting department: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Internal Server Error",
        )
