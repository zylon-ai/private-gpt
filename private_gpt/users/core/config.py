from functools import lru_cache
from typing import Any, Dict, Optional

from pydantic import PostgresDsn, validator
from pydantic_settings import BaseSettings


SQLALCHEMY_DATABASE_URI = "postgresql+psycopg2://{username}:{password}@{host}:{port}/{db_name}".format(
        host='localhost',
        port='5432',
        db_name='QuickGpt',
        username='postgres',
        password="quick",
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
    PORT: str


    # SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None

    # @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    # def assemble_db_connection(
    #     cls, v: Optional[str], values: Dict[str, Any]
    # ) -> Any:
    #     if isinstance(v, str):
    #         return v
    #     return PostgresDsn.build(
    #         scheme="postgresql",
    #         user=values.get("DB_USER"),
    #         password=values.get("DB_PASS"),
    #         host=values.get("DB_HOST"),
    #         path=f"/{values.get('DB_NAME') or  ''}",
    #     )
    # Database url configuration
    

    class Config:
        case_sensitive = True
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()