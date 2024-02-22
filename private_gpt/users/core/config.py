from functools import lru_cache
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings

SQLALCHEMY_DATABASE_URI = "postgresql+psycopg2://{username}:{password}@{host}:{port}/{db_name}".format(
    host='localhost',
    port='5432',
    db_name='QuickGpt',
    username='postgres',
    password="admin",
)

class Settings(BaseSettings):
    PROJECT_NAME: str = "AUTHENTICATION AND AUTHORIZATION"
    API_V1_STR: str = "/v1"
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    REFRESH_TOKEN_EXPIRE_MINUTES: int

    ENVIRONMENT: Optional[str]

    SUPER_ADMIN_EMAIL: str
    SUPER_ADMIN_PASSWORD: str
    SUPER_ADMIN_ACCOUNT_NAME: str

    DB_HOST: str
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str
    DB_PORT: str
    PORT: str

    SMTP_SERVER: str
    SMTP_PORT: str
    SMTP_SENDER_EMAIL: str
    SMTP_USERNAME: str
    SMTP_PASSWORD: str

    LDAP_SERVER: str
    LDAP_ENABLE: str

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        return f"postgresql+psycopg2://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    class Config:
        case_sensitive = True
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
