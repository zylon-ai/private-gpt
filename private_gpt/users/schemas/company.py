from typing import List
from pydantic import BaseModel

class CompanyBase(BaseModel):
    name: str


class CompanyCreate(CompanyBase):
    pass


class CompanyUpdate(CompanyBase):
    pass


class CompanyInDB(CompanyBase):
    id: int

    class Config:
        orm_mode = True


class Company(CompanyInDB):
    pass