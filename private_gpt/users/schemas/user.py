
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, EmailStr
from private_gpt.users.schemas.user_role import UserRole
from private_gpt.users.schemas.company import Company


class UserBaseSchema(BaseModel):
	email: EmailStr
	fullname: str
	company_id: int
	department_id: int

	class Config:
		arbitrary_types_allowed = True


class UserCreate(UserBaseSchema):
	password: str = Field(alias="password")


class UsernameUpdate(BaseModel):
	fullname: str


class UserUpdate(BaseModel):
	last_login: Optional[datetime] = None


class UserLoginSchema(BaseModel):
	email: EmailStr = Field(alias="email")
	password: str

	class Config:
		arbitrary_types_allowed = True


class UserSchema(UserBaseSchema):
	id: int
	user_role: Optional[UserRole] = None
	last_login: Optional[datetime] = None
	created_at: datetime
	updated_at: datetime
	is_active: bool = Field(default=False)

	class Config:
		orm_mode = True



class User(UserSchema):
    pass


class UserInDB(UserSchema):
    hashed_password: str


class Profile(UserBaseSchema):
	role: str


class DeleteUser(BaseModel):
	id: int


class UserAdminUpdate(BaseModel):
	id: int
	fullname: str
	role: str
	department_id: int
