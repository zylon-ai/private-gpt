from typing import Any, List

from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Depends, HTTPException, status, Security
from fastapi_pagination import Page, paginate

from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users import crud, models, schemas


router = APIRouter(prefix="/companies", tags=["Companies"])


@router.get("", response_model=Page[schemas.Company])
def list_companies(
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> Page[schemas.Company]:
    """
    Retrieve a list of companies with pagination support.
    """
    companies = crud.company.get_multi(db)
    return paginate(companies)


@router.post("/create", response_model=schemas.Company)
def create_company(
    company_in: schemas.CompanyCreate,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> schemas.Company:
    """
    Create a new company
    """
    company = crud.company.create(db=db, obj_in=company_in)
    company = jsonable_encoder(company)

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "message": "Company created successfully", 
            "subscription": company
        },
    )


@router.get("", response_model=Page[schemas.Company])
def list_companies(
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> Page[schemas.Company]:
    """
    Retrieve a list of companies with pagination support.
    """
    companies = crud.company.get_multi(db)
    return paginate(companies)

@router.get("/{company_id}", response_model=schemas.Company)
def read_company(
    company_id: int,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> schemas.Company:
    """
    Read a company by ID
    """
    company = crud.company.get_by_id(db, id=company_id)
    if company is None:
        raise HTTPException(status_code=404, detail="Company not found")
    return company


@router.put("/{company_id}", response_model=schemas.Company)
def update_company(
    company_id: int,
    company_in: schemas.CompanyUpdate,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> schemas.Company:
    """
    Update a company by ID
    """
    company = crud.company.get_by_id(db, id=company_id)
    if company is None:
        raise HTTPException(status_code=404, detail="Company not found")
    
    updated_company = crud.company.update(db=db, db_obj=company, obj_in=company_in)
    updated_company = jsonable_encoder(updated_company)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": f"{company_id} Company updated successfully", 
            "company": updated_company
        },
    )


@router.delete("/{company_id}", response_model=schemas.Company)
def delete_company(
    company_id: int,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> schemas.Company:
    """
    Delete a company by ID
    """
    
    company = crud.company.remove(db=db, id=company_id)
    if company is None:
        raise HTTPException(status_code=404, detail="Company not found")
    company = jsonable_encoder(company)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": "Company deleted successfully", 
            "company": company
        },
    )
