from typing import List
from datetime import datetime
from pydantic import BaseModel
from private_gpt.users.schemas.company import Company


class SubscriptionBase(BaseModel):
    start_date: datetime
    end_date: datetime
    is_active: bool

class SubscriptionCreate(SubscriptionBase):
    company_id: int

class SubscriptionSchema(SubscriptionBase):
    id: int
    company: Company

    class Config:
        orm_mode = True

class SubscriptionUpdate(SubscriptionBase):
    pass

class Subscription(SubscriptionBase):
    id: int
    company_id: int

    class Config:
        orm_mode = True
