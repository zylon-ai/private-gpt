from typing import List
from pydantic import BaseModel


class DepartmentBase(BaseModel):
    name: str


class DepartmentCreate(DepartmentBase):
    pass


class DepartmentUpdate(DepartmentBase):
    pass


class DepartmentInDB(DepartmentBase):
    id: int
    company_id: int

    class Config:
        orm_mode = True

class DepartmentAdminCreate(DepartmentBase):
    company_id: int

    class Config:
        orm_mode = True

class Department(DepartmentInDB):
    pass
