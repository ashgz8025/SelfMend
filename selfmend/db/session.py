"""
Database session management for SelfMend.

Uses SQLAlchemy 2.0 with PostgreSQL (via DATABASE_URL env var).
Falls back to SQLite for development when Postgres is not available.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

_engine = None
_SessionLocal = None


def _get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    if not url or "REPLACE" in url:
        # Fallback to SQLite
        sqlite_path = ROOT / "data" / "selfmend.db"
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{sqlite_path}"
        log.info("DATABASE_URL not set — using SQLite at %s", sqlite_path)
    return url


def get_engine():
    global _engine
    if _engine is None:
        url = _get_database_url()
        kwargs: dict = {"echo": False}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        _engine = create_engine(url, **kwargs)
        Base.metadata.create_all(_engine)
        log.info("Database engine ready: %s", url.split("@")[-1] if "@" in url else url)
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SessionLocal


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager that yields a database session and handles commit/rollback."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
