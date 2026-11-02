"""Deterministic release rules from rules.yaml.

The LLM only explains; this module decides GREEN / YELLOW / RED.
First matching rule wins, and the report shows which rule fired.
"""
from __future__ import annotations

import yaml

from .config import rules_path

RULE_DESCRIPTIONS = {
    "critical_real_bug": "RED: a Real product bug failure in a test tagged critical.",
    "pass_rate_below_90": "RED: pass rate below 90%.",
    "only_flaky_or_environment": "YELLOW: every failure is a flaky or environment problem.",
    "pass_rate_90_97": "YELLOW: pass rate between 90% and 97%.",
    "all_clear": "GREEN: no rule for RED or YELLOW fired.",
}


def load_rules(path=None) -> dict:
    with open(path or rules_path(), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def is_critical(test_name: str, critical_property: bool, rules: dict | None = None) -> bool:
    if critical_property:
        return True
    rules = rules or load_rules()
    name = test_name.lower()
    return any(kw.lower() in name for kw in rules.get("critical_keywords", []))


def decide(pass_rate: float, failures: list[dict], rules: dict | None = None) -> tuple[str, str]:
    """Return (verdict, rule_key). failures: dicts with category + critical."""
    rules = rules or load_rules()
    red_below = float(rules.get("red_pass_rate_below", 90))
    yellow_low = float(rules.get("yellow_pass_rate_low", 90))
    yellow_high = float(rules.get("yellow_pass_rate_high", 97))
    if any(f.get("category") == "real_bug" and f.get("critical") for f in failures):
        return "RED", "critical_real_bug"
    if pass_rate < red_below:
        return "RED", "pass_rate_below_90"
    if failures and all(f.get("category") in ("flaky", "environment") for f in failures):
        return "YELLOW", "only_flaky_or_environment"
    if yellow_low <= pass_rate < yellow_high:
        return "YELLOW", "pass_rate_90_97"
    return "GREEN", "all_clear"


def rule_description(rule_key: str) -> str:
    return RULE_DESCRIPTIONS.get(rule_key, rule_key)
