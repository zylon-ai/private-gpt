from datetime import datetime
from sqlalchemy.orm import Session
from private_gpt.users.models.audit import Audit

def log_audit_entry(
    session: Session,
    model: str,
    action: str,
    details: dict,
    user_id: int = None,
    ip_address: str = None,
):
    audit_entry = Audit(
        timestamp=datetime.utcnow(),
        user_id=user_id,
        model=model,
        action=action,
        details=details,
        ip_address=ip_address
    )
    
    session.add(audit_entry)
    session.commit()



