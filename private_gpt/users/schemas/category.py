from pydantic import BaseModel
from typing import Optional, List

class CategoryBase(BaseModel):
    name: str

class CategoryCreate(CategoryBase):
    pass

class CategoryUpdate(CategoryBase):
    id: int
    name: Optional[str] = None

class CategoryInDBBase(CategoryBase):
    id: int

    class Config:
        orm_mode = True

class Category(CategoryInDBBase):
    pass

class CategoryInDB(CategoryInDBBase):
    pass

class CategoryList(BaseModel):
    id: int
    name: str

class CategoryDelete(BaseModel):
    id: int

