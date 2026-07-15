"""
SAFER Data Service — Database Engine

SQLAlchemy setup with pluggable backend:
  - Development: SQLite  (zero-config)
  - Production:  PostgreSQL (swap via DATABASE_URL env)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from shared.config import DATABASE_URL

# ─── Engine ─────────────────────────────────────────────────────────────────

_connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    _connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    echo=False,  # Set True for SQL debugging
    pool_pre_ping=True,
)

# Enable WAL mode for SQLite (better concurrent read performance)
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

# ─── Session ────────────────────────────────────────────────────────────────

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session and auto-closes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─── Base ───────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


def init_db():
    """Create all tables. Safe to call multiple times (CREATE IF NOT EXISTS)."""
    from data_service import models  # noqa: F401 — force model registration
    Base.metadata.create_all(bind=engine)
    print("[Data Service] Database tables initialized.")
