"""Seed 20 historical runs into SQLite (synthetic data).

Test pool: 10 tests that always pass, 8 flaky tests (pass ~60% of runs),
4 tests that fail often recently, 3 tests that fail sometimes. This
gives the flaky query and the re-check ranking something real to find.
Deterministic (seed 123). Seeding is idempotent: if runs already exist,
nothing is added.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from .db import Failure, Run, TestResult

ALWAYS_PASS = [
    "test_login_page_loads", "test_search_basic", "test_homepage_title",
    "test_logout", "test_help_page", "test_footer_links",
    "test_about_page", "test_privacy_page", "test_terms_page", "test_sitemap",
]
FLAKY = [
    "test_dashboard_load", "test_modal_open", "test_table_refresh",
    "test_chart_render", "test_filter_apply", "test_row_click",
    "test_menu_select", "test_animation_done",
]
OFTEN_FAIL = [
    "test_checkout_total", "test_login_valid_user",
    "test_api_health", "test_db_connect",
]
SOMETIMES_FAIL = [
    "test_import_users", "test_cart_total", "test_search_results",
]

POOL = ALWAYS_PASS + FLAKY + OFTEN_FAIL + SOMETIMES_FAIL

# (category, message) used for seeded failure rows.
FAILURE_KIND = {
    "test_checkout_total": ("real_bug", "AssertionError: expected 59.99 but got 49.99"),
    "test_login_valid_user": ("real_bug", "AssertionError: expected 200 but got 401 for valid credentials"),
    "test_api_health": ("environment", "Connection refused to api.internal:8080"),
    "test_db_connect": ("environment", "Connection refused: could not connect to database at db.internal:5432"),
    "test_dashboard_load": ("flaky", "Timeout: element not visible after 10s wait, passed on retry"),
    "test_modal_open": ("flaky", "Timeout waiting for modal, succeeded on second attempt"),
    "test_table_refresh": ("flaky", "Element click intercepted, passed when re-run alone"),
    "test_chart_render": ("flaky", "Timeout waiting for chart animation, passed on retry"),
    "test_filter_apply": ("flaky", "Flaky timeout: filter results arrived late, passed on rerun"),
    "test_row_click": ("flaky", "StaleElementReferenceException: element is not attached to the page"),
    "test_menu_select": ("flaky", "Stale element after page refresh, passed on retry"),
    "test_animation_done": ("flaky", "Timing sensitive: assertion ran before animation finished"),
    "test_import_users": ("test_data", "FileNotFoundError: fixture users.csv not found"),
    "test_cart_total": ("real_bug", "AssertionError: expected 59.99 but got 49.99"),
    "test_search_results": ("real_bug", "AttributeError: 'NoneType' object has no attribute 'items'"),
}


def _should_fail(rng: random.Random, name: str, run_index: int) -> bool:
    if name in ALWAYS_PASS:
        return False
    if name in FLAKY:
        return rng.random() < 0.4
    if name in OFTEN_FAIL:
        # Fail more in recent runs so re-check ranking surfaces them.
        return rng.random() < (0.7 if run_index >= 12 else 0.3)
    return rng.random() < 0.25


def seed_historical_runs(session, n: int = 20) -> int:
    """Seed n historical runs. Returns number of runs added (0 if seeded)."""
    if session.query(Run).count() > 0:
        return 0
    rng = random.Random(123)
    now = datetime.utcnow()
    for i in range(n):
        created = now - timedelta(days=(n - i))
        run = Run(created_at=created, source="seed", total=0, passed=0, failed=0,
                  verdict="", pass_rate=100.0)
        session.add(run)
        session.flush()
        passed = failed = 0
        for name in POOL:
            if rng.random() < 0.05:
                status = "skipped"
            elif _should_fail(rng, name, i):
                status = "failed"
            else:
                status = "passed"
            tr = TestResult(run_id=run.id, test_name=name, status=status)
            session.add(tr)
            session.flush()
            if status == "passed":
                passed += 1
            elif status == "failed":
                failed += 1
                category, message = FAILURE_KIND.get(
                    name, ("flaky", "Flaky timeout: passed on retry"))
                session.add(Failure(test_result_id=tr.id, test_name=name,
                                    category=category, confidence=0.7,
                                    message=message, log_text=message))
        run.total = passed + failed
        run.passed = passed
        run.failed = failed
        run.pass_rate = 100.0 * passed / (passed + failed) if (passed + failed) else 100.0
        run.verdict = "GREEN" if run.pass_rate >= 97 else ("YELLOW" if run.pass_rate >= 90 else "RED")
    session.commit()
    return n
