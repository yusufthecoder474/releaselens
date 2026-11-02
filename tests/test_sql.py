"""SQL tests use their own in-memory SQLite database (not the app's),
so they are independent of seeding and of other tests.
"""
from datetime import datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base, Run, TestResult
from app.flaky import flaky_tests, recheck_first
from app.flaky import test_history as history_for_test


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.close()


def _add_run(session, day_offset: int, results: dict[str, str]) -> int:
    run = Run(created_at=datetime.utcnow() - timedelta(days=day_offset),
              source="test", total=len(results),
              passed=sum(1 for v in results.values() if v == "passed"),
              failed=sum(1 for v in results.values() if v == "failed"),
              verdict="", pass_rate=100.0)
    session.add(run)
    session.flush()
    for name, status in results.items():
        session.add(TestResult(run_id=run.id, test_name=name, status=status))
    session.commit()
    return run.id


def test_flaky_query_finds_both_outcomes(session):
    for i in range(4):
        _add_run(session, i, {"test_a": "passed" if i % 2 == 0 else "failed",
                              "test_b": "passed",
                              "test_c": "failed"})
    assert flaky_tests(session) == ["test_a"]
    assert "test_b" not in flaky_tests(session)
    assert "test_c" not in flaky_tests(session)


def test_recheck_ranks_by_failure_rate(session):
    for i in range(4):
        _add_run(session, i, {"test_high": "failed",
                              "test_mid": "failed" if i < 2 else "passed",
                              "test_ok": "passed"})
    ranked = recheck_first(session)
    assert ranked[0]["test_name"] == "test_high"
    assert ranked[0]["failure_rate"] == 1.0
    assert ranked[1]["test_name"] == "test_mid"
    assert ranked[1]["failure_rate"] == 0.5


def test_history_newest_first(session):
    _add_run(session, 2, {"test_a": "passed"})
    _add_run(session, 1, {"test_a": "failed"})
    hist = history_for_test(session, "test_a")
    assert [h["status"] for h in hist] == ["failed", "passed"]
