"""SQL queries over run history: flaky tests and tests to re-check first.

Both are real SQL (SQLAlchemy GROUP BY / HAVING / ORDER BY) over the
last 10 runs, not Python-side aggregation.
"""
from __future__ import annotations

from sqlalchemy import case, func

from .db import Run, TestResult

LAST_N_RUNS = 10


def last_run_ids(session, n: int = LAST_N_RUNS) -> list[int]:
    rows = session.query(Run.id).order_by(Run.id.desc()).limit(n).all()
    return [r[0] for r in rows]


def flaky_tests(session, n: int = LAST_N_RUNS) -> list[str]:
    """Tests with BOTH passed and failed results in the last n runs."""
    run_ids = last_run_ids(session, n)
    if not run_ids:
        return []
    passed_expr = func.sum(case((TestResult.status == "passed", 1), else_=0))
    failed_expr = func.sum(case((TestResult.status == "failed", 1), else_=0))
    rows = (
        session.query(TestResult.test_name,
                      passed_expr.label("n_pass"), failed_expr.label("n_fail"))
        .filter(TestResult.run_id.in_(run_ids))
        .group_by(TestResult.test_name)
        .having(passed_expr > 0)
        .having(failed_expr > 0)
        .order_by(TestResult.test_name)
        .all()
    )
    return [r[0] for r in rows]


def recheck_first(session, n: int = LAST_N_RUNS, limit: int = 10) -> list[dict]:
    """Rank tests by recent failure rate (failures / appearances).

    Ordered by failure rate desc, then failure count desc, then name.
    Only tests seen in the last n runs are ranked.
    """
    run_ids = last_run_ids(session, n)
    if not run_ids:
        return []
    appearances_expr = func.count(TestResult.id)
    failures_expr = func.sum(case((TestResult.status == "failed", 1), else_=0))
    rows = (
        session.query(TestResult.test_name,
                      appearances_expr.label("appearances"),
                      failures_expr.label("failures"))
        .filter(TestResult.run_id.in_(run_ids))
        .group_by(TestResult.test_name)
        .having(failures_expr > 0)
        .order_by((failures_expr * 1.0 / appearances_expr).desc(),
                  failures_expr.desc(), TestResult.test_name)
        .limit(limit)
        .all()
    )
    out = []
    for name, appearances_n, failures_n in rows:
        out.append({
            "test_name": name,
            "failure_rate": round(failures_n / appearances_n, 3) if appearances_n else 0.0,
            "failures": int(failures_n),
            "appearances": int(appearances_n),
        })
    return out


def failure_rates(session, test_names: list[str], n: int = LAST_N_RUNS) -> dict[str, float]:
    """Recent failure rate per test over the last n runs.

    Returns 0.0 for tests with no history in the window. Used by the
    deterministic "tests to run first" ranking (see selection.py).
    """
    run_ids = last_run_ids(session, n)
    if not run_ids or not test_names:
        return {name: 0.0 for name in test_names}
    appearances_expr = func.count(TestResult.id)
    failures_expr = func.sum(case((TestResult.status == "failed", 1), else_=0))
    rows = (
        session.query(TestResult.test_name,
                      appearances_expr.label("appearances"),
                      failures_expr.label("failures"))
        .filter(TestResult.run_id.in_(run_ids), TestResult.test_name.in_(test_names))
        .group_by(TestResult.test_name)
        .all()
    )
    rates = {name: 0.0 for name in test_names}
    for name, appearances_n, failures_n in rows:
        rates[name] = round(failures_n / appearances_n, 3) if appearances_n else 0.0
    return rates


def test_history(session, test_name: str, n: int = LAST_N_RUNS) -> list[dict]:
    """Most recent results for one test, newest first."""
    run_ids = last_run_ids(session, n)
    if not run_ids:
        return []
    rows = (
        session.query(TestResult.run_id, TestResult.status, Run.created_at)
        .join(Run, Run.id == TestResult.run_id)
        .filter(TestResult.run_id.in_(run_ids), TestResult.test_name == test_name)
        .order_by(TestResult.run_id.desc())
        .all()
    )
    return [{"run_id": r[0], "status": r[1], "created_at": r[2].isoformat() if r[2] else ""} for r in rows]
