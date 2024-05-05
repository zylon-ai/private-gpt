from typing import Optional, List

from private_gpt.users.crud.base import CRUDBase
from private_gpt.users.models.subscription import Subscription
from private_gpt.users.schemas.subscription import SubscriptionCreate, SubscriptionUpdate
from sqlalchemy.orm import Session
from datetime import datetime

class CRUDSubscription(CRUDBase[Subscription, SubscriptionCreate, SubscriptionUpdate]):

    def get_by_id(self, db: Session, *, subscription_id: int) -> Optional[Subscription]:
        return db.query(self.model).filter(Subscription.sub_id == subscription_id).first()
    
    def get_by_company_id(self, db: Session, *, company_id: int) -> List[Subscription]:
        return db.query(self.model).filter(Subscription.company_id == company_id).all()
    
    def get_active_subscription_by_company(self, db: Session, *, company_id: int) -> List[Subscription]:
        current_datetime = datetime.utcnow()
        return (
            db.query(self.model)
            .filter(
                Subscription.company_id == company_id,
                Subscription.is_active == True,  # Active subscriptions
                Subscription.end_date >= current_datetime,  # End date is not passed
            )
            .all()
        )

subscription = CRUDSubscription(Subscription)


