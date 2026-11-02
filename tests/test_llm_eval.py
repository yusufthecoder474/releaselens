"""Cached LLM evaluation loader tests (fake LLM, no key, no network).

Covers: cache round-trip, cache reuse (second run makes no new LLM
calls), invalid-category counting with fallback, and grounding-rate
computation for verbatim vs invented evidence lines.
"""
import json
from pathlib import Path

import pytest

from scripts.llm_evaluate import (grounding_rate_raw, load_cache, run_c,
                                  save_cache)


class _FakeCategory:
    def __init__(self, value: str) -> None:
        self.value = value


class _FakeExplanation:
    def __init__(self, category: str, evidence: list[str]) -> None:
        self.category = _FakeCategory(category)
        self.plain_english_reason = "Fake reason."
        self.evidence_lines = evidence
        self.suggested_next_step = "Fake step."


class FakeLLM:
    """Deterministic stand-in for ChatOpenAI."""

    def __init__(self, category: str = "flaky", evidence_mode: str = "verbatim") -> None:
        self.category = category
        self.evidence_mode = evidence_mode
        self.calls = 0

    def with_structured_output(self, cls):
        outer = self

        class _Structured:
            def invoke(self, prompt: str):
                outer.calls += 1
                if outer.evidence_mode == "verbatim":
                    # First non-empty line of the Log: section.
                    log_part = prompt.split("Log:\n", 1)[1] if "Log:\n" in prompt else ""
                    lines = [ln.strip() for ln in log_part.splitlines() if ln.strip()]
                    evidence = lines[:1]
                elif outer.evidence_mode == "invented":
                    evidence = ["this line was never in any log"]
                else:
                    evidence = []
                return _FakeExplanation(outer.category, evidence)

        return _Structured()

    def invoke(self, prompt: str):
        class _Reply:
            content = "Final answer: done"
        return _Reply()


def _subset_rows():
    """10 rows: 5 families x 2 rows, so GroupKFold(5) works."""
    from app.classifier import load_labeled_rows

    by_family: dict[str, list[dict]] = {}
    for r in load_labeled_rows():
        by_family.setdefault(r["template_family"], []).append(r)
    rows = []
    for fam in sorted(by_family)[:5]:
        rows.extend(by_family[fam][:2])
    assert len(rows) == 10
    return rows


def test_cache_round_trip(tmp_path):
    path = tmp_path / "cache.json"
    assert load_cache(path) == {}
    save_cache({"k": {"category": "flaky"}}, path)
    assert load_cache(path) == {"k": {"category": "flaky"}}


def test_grounding_rate_raw():
    log = "First line\nSecond line"
    assert grounding_rate_raw(["First line"], log) == (1, 1)
    assert grounding_rate_raw(["Invented line"], log) == (0, 1)
    assert grounding_rate_raw([], log) == (0, 0)


def test_run_c_caches_and_reuses(tmp_path):
    rows = _subset_rows()
    cache_path = tmp_path / "cache.json"
    cache: dict = {}
    fake = FakeLLM(category="flaky", evidence_mode="verbatim")
    m1 = run_c(rows, fake, "v2", "fake-model", cache, cache_path)
    calls_first = fake.calls
    assert calls_first == 10
    assert m1["n_items"] == 10
    # Second run with the saved cache: no new LLM calls, same metrics.
    cache2 = load_cache(cache_path)
    fake2 = FakeLLM(category="flaky", evidence_mode="verbatim")
    m2 = run_c(rows, fake2, "v2", "fake-model", cache2, cache_path)
    assert fake2.calls == 0
    assert m2["accuracy"] == m1["accuracy"]
    assert m2["macro_f1"] == m1["macro_f1"]


def test_run_c_counts_invalid_category(tmp_path):
    rows = _subset_rows()
    cache_path = tmp_path / "cache.json"
    fake = FakeLLM(category="bogus", evidence_mode="verbatim")
    m = run_c(rows, fake, "v2", "fake-model", {}, cache_path)
    assert m["invalid_count"] == 10
    assert m["invalid_category_rate"] == 1.0
