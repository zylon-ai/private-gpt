from typing import List
from datetime import datetime
from pydantic import BaseModel

class CompanyBase(BaseModel):
    name: str

class CompanyCreate(CompanyBase):
    pass

class CompanyUpdate(CompanyBase):
    pass


class Company(CompanyBase):
    id: int
    
    class Config:
        orm_mode = True