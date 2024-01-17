from typing import Optional, List

from private_gpt.users.crud.base import CRUDBase
from private_gpt.users.models.subscription import Subscription
from private_gpt.users.schemas.subscription import SubscriptionCreate, SubscriptionUpdate
from sqlalchemy.orm import Session


class CRUDSubscription(CRUDBase[Subscription, SubscriptionCreate, SubscriptionUpdate]):
    def get_by_id(self, db: Session, *, subscription_id: int) -> Optional[Subscription]:
        return db.query(self.model).filter(Subscription.sub_id == subscription_id).first()
    
    def get_by_company_id(self, db: Session, *, company_id: int) -> List[Subscription]:
        return db.query(self.model).filter(Subscription.company_id == company_id).all()

subscription = CRUDSubscription(Subscription)