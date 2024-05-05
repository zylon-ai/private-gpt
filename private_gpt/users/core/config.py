from typing import Optional
from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "QuickGPT"
    API_V1_STR: str = "/v1"
    SECRET_KEY: str
    REFRESH_KEY: str

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
    LDAP_ENABLE: bool
    ENABLE_MAKER_CHECKER: bool
    
    class Config:
        case_sensitive = True
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
