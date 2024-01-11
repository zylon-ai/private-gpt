from typing import Union, Any, Generator
from datetime import datetime
from private_gpt.users import crud, models, schemas
from private_gpt.users.db.session import SessionLocal
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from private_gpt.users.core.security import (
    ALGORITHM,
    JWT_SECRET_KEY
)
from fastapi import Depends, HTTPException, Security, status
from jose import jwt
from pydantic import ValidationError
from private_gpt.users.schemas.token import TokenPayload
from sqlalchemy.orm import Session

reuseable_oauth = OAuth2PasswordBearer(
    tokenUrl="/login",
    scheme_name="JWT"
)

def get_db() -> Generator:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

async def get_current_user(
        db: Session = Depends(get_db), 
        token: str = Depends(reuseable_oauth)
    ) -> models.User:

    try:
        payload = jwt.decode(
            token, JWT_SECRET_KEY, algorithms=[ALGORITHM]
        )
        token_data = TokenPayload(**payload)
        
        if datetime.fromtimestamp(token_data.exp) < datetime.now():
            raise HTTPException(
                status_code = status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except(jwt.JWTError, ValidationError):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    user: Union[dict[str, Any], None] = crud.user.get(db, id=token_data.id)
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Could not find user",
        )
    
    return user

async def get_current_active_user(
    current_user: models.User = Security(get_current_user, scopes=[],),
) -> models.User:
    if not crud.user.is_active(current_user):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user