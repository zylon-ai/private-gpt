from pydantic import BaseModel


class TokenSchema(BaseModel):
    access_token: str
    refresh_token: str
    user: object

    class Config:
        arbitrary_types_allowed = True
    
class TokenPayload(BaseModel):
    id: int
    role: str = None
    company: str = None

    class Config:
        arbitrary_types_allowed = True