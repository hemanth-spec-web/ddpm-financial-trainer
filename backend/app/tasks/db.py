"""
Shared synchronous database engine for Celery tasks.
Separated into its own module to avoid circular imports between
training_tasks.py and generation_tasks.py (both need this engine).
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

SYNC_DATABASE_URL = settings.DATABASE_URL.replace("postgresql+asyncpg", "postgresql+psycopg2")

sync_engine = create_engine(SYNC_DATABASE_URL)
SyncSession = sessionmaker(bind=sync_engine)