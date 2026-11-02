"""Fixed failure categories. The classifier, the LLM output enum, the
verify guard and the report all use these exact values — nothing else
may appear as a category.
"""
from __future__ import annotations

from enum import Enum


class Category(str, Enum):
    real_bug = "real_bug"
    environment = "environment"
    flaky = "flaky"
    test_data = "test_data"
    dependency = "dependency"


LABELS: dict[str, str] = {
    "real_bug": "Real product bug",
    "environment": "Server/environment problem",
    "flaky": "Flaky or timing issue",
    "test_data": "Test data or configuration problem",
    "dependency": "Dependency or build problem",
}

NEXT_STEPS: dict[str, str] = {
    "real_bug": "Send this to the owning engineer to fix before release, and add a regression test for it.",
    "environment": "Check the server, service or network it needs, then re-run the test once the environment is healthy.",
    "flaky": "Re-run the test a few times. If it passes sometimes and fails sometimes, treat it as flaky and quarantine it.",
    "test_data": "Fix the test data or configuration (fixture, config value, seed data) and re-run.",
    "dependency": "Fix the dependency or build step (install, version, build error) and re-run.",
}

ALL_CATEGORIES: list[str] = [c.value for c in Category]


def is_valid_category(value: str) -> bool:
    return value in LABELS
