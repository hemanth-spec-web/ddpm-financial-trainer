from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import Optional


class Settings(BaseSettings):
    DATABASE_URL: str
    REDIS_URL: str = "redis://localhost:6379/0"
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    FRONTEND_URL: str = "http://localhost:5173"
    USE_CELERY: bool = True

    @field_validator("DATABASE_URL")
    @classmethod
    def fix_async_driver(cls, v: str) -> str:
        """
        Render (and some other hosts) provide DATABASE_URL as a plain
        postgres:// or postgresql:// URL, which defaults to the sync
        psycopg2 driver. Our app uses SQLAlchemy's async engine, which
        requires the asyncpg driver explicitly in the URL scheme.
        """
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+asyncpg://", 1)
        if v.startswith("postgresql://") and "+asyncpg" not in v:
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()