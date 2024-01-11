from pydantic import BaseModel


class TokenSchema(BaseModel):
    access_token: str
    refresh_token: str

    class Config:
        arbitrary_types_allowed = True
    
class TokenPayload(BaseModel):
    id: int
    role: str = None

    class Config:
        arbitrary_types_allowed = True