from typing import Any, List

from sqlalchemy.orm import Session
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter, Depends, HTTPException, status, Security
from fastapi_pagination import Page, paginate

from private_gpt.users.api import deps
from private_gpt.users.constants.role import Role
from private_gpt.users import crud, models, schemas

router = APIRouter(prefix="/categories", tags=["Categories"])


@router.get("", response_model=Page[schemas.Category])
def list_categories(
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[
            Role.ADMIN["name"],
            Role.SUPER_ADMIN["name"], 
            Role.OPERATOR["name"]
            ],
    ),
) -> Page[schemas.Category]:
    """
    Retrieve a list of categories with pagination support.
    """
    categories = crud.category.get_multi(db)
    return paginate(categories)


@router.post("/create", response_model=schemas.Category)
def create_category(
    category_in: schemas.CategoryCreate,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
         scopes=[
            Role.SUPER_ADMIN["name"], 
            Role.OPERATOR["name"]
        ],
    ),
) -> schemas.Category:
    """
    Create a new category.
    """
    category = crud.category.create(db=db, obj_in=category_in)
    category = jsonable_encoder(category)

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "message": "Category created successfully",
            "category": category,
        },
    )


@router.get("/{category_id}", response_model=schemas.Category)
def read_category(
    category_id: int,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[
            Role.ADMIN["name"],
            Role.SUPER_ADMIN["name"], 
            Role.OPERATOR["name"]
            ],
    ),
) -> schemas.Category:
    """
    Read a category by ID.
    """
    category = crud.category.get_by_id(db, id=category_id)
    if category is None:
        raise HTTPException(status_code=404, detail="Category not found")
    return category


@router.put("/update", response_model=schemas.Category)
def update_category(
    category_in: schemas.CategoryUpdate,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[
            Role.SUPER_ADMIN["name"], 
            Role.OPERATOR["name"]
            ],
    ),
) -> schemas.Category:
    """
    Update a category by ID.
    """
    category = crud.category.get_by_id(db, id=category_in.id)
    if category is None:
        raise HTTPException(status_code=404, detail="Category not found")
    
    updated_category = crud.category.update(db=db, db_obj=category, obj_in=category_in)
    updated_category = jsonable_encoder(updated_category)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": f"{category_in.id} Category updated successfully",
            "category": updated_category,
        },
    )


@router.delete("/delete", response_model=schemas.Category)
def delete_category(
    category_in: schemas.CategoryDelete,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Security(
        deps.get_current_user,
        scopes=[Role.SUPER_ADMIN["name"]],
    ),
) -> schemas.Category:
    """
    Delete a category by ID.
    """
    category_id = category_in.id
    category = crud.category.remove(db=db, id=category_id)
    if category is None:
        raise HTTPException(status_code=404, detail="Category not found")
    category = jsonable_encoder(category)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "message": "Category deleted successfully",
            "category": category,
        },
    )
