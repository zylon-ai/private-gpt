from typing import List
from pydantic import BaseModel

from private_gpt.users.schemas import Department, User


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
    subscriptions: List[str] = []
    users: List[User] = []
    user_roles: List[str] = []
    departments: List[Department] = []
