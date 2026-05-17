"""
app/database.py
─────────────────────────────────────────────────────────
SQLAlchemy ORM models + SQLite session management.
Database file: ./spoof_detection.db (project root)

Fixes applied:
  - db.close() was called twice in get_db() finally block  ← BUG FIXED
  - declarative_base from deprecated sqlalchemy.ext.declarative  ← FIXED
  - Removed unused Session import
"""

import datetime
from pathlib import Path

from sqlalchemy import (
    Boolean, Column, DateTime, Float,
    Integer, String, Text, create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker   # correct in SQLAlchemy 2.x

# ── Database location ──────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
DB_PATH      = BASE_DIR / "spoof_detection.db"
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},   # required for SQLite + FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── ORM Models ─────────────────────────────────────────────────────────────────

class User(Base):
    """Registered user with face embeddings for identity verification."""
    __tablename__ = "users"

    id              = Column(Integer, primary_key=True, index=True)
    username        = Column(String(64),  unique=True, index=True, nullable=False)
    email           = Column(String(128), unique=True, index=True, nullable=False)
    hashed_password = Column(String(256), nullable=False)

    # JSON-serialized list of embedding vectors → list[list[float]]
    # Multiple enrollment sessions accumulate here.
    face_embeddings = Column(Text, default="[]", nullable=False)

    registered_at   = Column(DateTime, default=datetime.datetime.utcnow)
    last_login      = Column(DateTime, nullable=True)
    is_active       = Column(Boolean, default=True)

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r}>"


class SpoofLog(Base):
    """Audit log for every verification attempt."""
    __tablename__ = "spoof_logs"

    id               = Column(Integer, primary_key=True, index=True)
    user_id          = Column(Integer, nullable=True)    # NULL = unknown / anonymous
    timestamp        = Column(DateTime, default=datetime.datetime.utcnow)

    # Per-model verdicts
    depth_verdict    = Column(String(8),  nullable=True)   # "2D" | "3D"
    depth_std        = Column(Float,      nullable=True)
    clip_verdict     = Column(String(16), nullable=True)   # "real" | "spoof"
    clip_confidence  = Column(Float,      nullable=True)
    yolo_person_conf = Column(Float,      nullable=True)

    # Final scores
    combined_score   = Column(Float,   nullable=True)    # 0-1, higher = more real
    is_spoof         = Column(Boolean, nullable=True)
    face_matched     = Column(Boolean, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<SpoofLog id={self.id} user_id={self.user_id} "
            f"is_spoof={self.is_spoof} ts={self.timestamp}>"
        )


# ── Helpers ────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables (idempotent — safe to call on every startup)."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session and closes it exactly once."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()    # ← called only once (was incorrectly called twice before)