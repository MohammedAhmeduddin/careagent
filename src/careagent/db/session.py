"""
careagent.db.session
~~~~~~~~~~~~~~~~~~~~
Database engine and session factory.
"""

from collections.abc import Generator
from contextlib import contextmanager

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from careagent.config import get_settings
from careagent.db.models import Base

settings = get_settings()

engine = create_engine(
    settings.database_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=settings.log_level.upper() == "DEBUG",
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def create_tables() -> None:
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    # create_all never alters existing tables; add columns introduced later
    with engine.begin() as conn:
        for statement in MIGRATIONS:
            conn.execute(text(statement))
    logger.info("Tables ready.")


LEGACY_TABLES = ["provider_services"]

# Idempotent column additions for databases created by an earlier version
MIGRATIONS = [
    "ALTER TABLE providers ADD COLUMN IF NOT EXISTS narrative_fingerprint VARCHAR(64)",
]


def reset_schema() -> None:
    """Drop and recreate all tables, including tables from the old schema."""
    logger.warning("Dropping all CareAgent tables...")
    with engine.begin() as conn:
        for table in LEGACY_TABLES:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        for table in reversed(Base.metadata.sorted_tables):
            conn.execute(text(f"DROP TABLE IF EXISTS {table.name} CASCADE"))
    create_tables()


def check_connection() -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error(f"Database connection failed: {exc}")
        return False


@contextmanager
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
