"""SQLite via SQLAlchemy: runs, test_results, failures.

Engine/session are created from the current RELEASELENS_DB env var so
tests can point at a temp database. Call init_db() once at startup (or
lazily on first use) and seed_historical_runs() to fill history.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import database_url


class Base(DeclarativeBase):
    pass


class Run(Base):
    __test__ = False
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    source: Mapped[str] = mapped_column(String(200), default="")
    total: Mapped[int] = mapped_column(Integer, default=0)
    passed: Mapped[int] = mapped_column(Integer, default=0)
    failed: Mapped[int] = mapped_column(Integer, default=0)
    verdict: Mapped[str] = mapped_column(String(10), default="")
    pass_rate: Mapped[float] = mapped_column(Float, default=100.0)


class TestResult(Base):
    __test__ = False
    __tablename__ = "test_results"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(Integer, ForeignKey("runs.id"), index=True)
    test_name: Mapped[str] = mapped_column(String(300), index=True)
    status: Mapped[str] = mapped_column(String(20))  # passed | failed | skipped


class Failure(Base):
    __test__ = False
    __tablename__ = "failures"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    test_result_id: Mapped[int] = mapped_column(Integer, ForeignKey("test_results.id"), index=True)
    test_name: Mapped[str] = mapped_column(String(300), index=True)
    category: Mapped[str] = mapped_column(String(50))
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    log_text: Mapped[str] = mapped_column(Text, default="")


_engine = None
_Session = None


def get_engine():
    global _engine, _Session
    if _engine is None:
        _engine = create_engine(database_url(), connect_args={"check_same_thread": False})
        _Session = sessionmaker(bind=_engine)
    return _engine


def get_session():
    get_engine()
    assert _Session is not None
    return _Session()


def init_db() -> None:
    engine = get_engine()
    Base.metadata.create_all(engine)


def reset_engine() -> None:
    """For tests: drop the cached engine so a new env var takes effect."""
    global _engine, _Session
    _engine = None
    _Session = None
