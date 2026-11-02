"""Deterministic "tests to run first" ranking (no LLM involved).

Given an optional list of changed file paths submitted with the upload,
rank the tests in the current run by:
  (a) name/path keyword overlap with the changed files,
  (b) recent failure rate from SQL history (last 10 runs),
  (c) flaky status (a bonus if the test has both passed and failed).

Score (documented, fixed weights):
    score = 2 * overlap_tokens + 5 * failure_rate + (1 if flaky else 0)
Ties break by failure rate desc, then test name asc (deterministic).

Overlap tokens: lowercase alphanumeric tokens from splitting changed
paths on '/', '.', '_', '-' and from splitting test names on '_'.
A token counts once per test even if it appears in several changed
files. Common filler tokens (test, tests, py, js, ts, src) are ignored
so "test_foo.py" does not match every test.

This is a heuristic: it ranks by surface similarity and history, not
by any real dependency analysis. See LIMITATIONS.md.
"""
from __future__ import annotations

import re

FILLER_TOKENS = {"test", "tests", "py", "js", "ts", "tsx", "jsx", "src", "spec"}
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in FILLER_TOKENS}


def changed_file_tokens(changed_files: list[str]) -> set[str]:
    tokens: set[str] = set()
    for path in changed_files:
        tokens |= tokenize(path.replace("/", "_"))
    return tokens


def overlap_count(test_name: str, file_tokens: set[str]) -> int:
    return len(tokenize(test_name) & file_tokens)


def rank_tests(tests: list[str], changed_files: list[str],
               failure_rates: dict[str, float], flaky: set[str],
               limit: int = 10) -> list[dict]:
    """Rank tests. failure_rates maps test name -> 0..1 failure rate
    (last 10 runs); flaky is the set of flaky test names."""
    file_tokens = changed_file_tokens(changed_files)
    ranked = []
    for name in tests:
        overlap = overlap_count(name, file_tokens)
        rate = float(failure_rates.get(name, 0.0))
        is_flaky = name in flaky
        score = 2 * overlap + 5 * rate + (1 if is_flaky else 0)
        ranked.append({
            "test_name": name,
            "score": round(score, 3),
            "overlap": overlap,
            "failure_rate": round(rate, 3),
            "flaky": is_flaky,
        })
    ranked.sort(key=lambda r: (-r["score"], -r["failure_rate"], r["test_name"]))
    return ranked[:limit]


def parse_changed_files(raw: str | None) -> list[str]:
    """Parse a comma- or newline-separated changed-files field."""
    if not raw:
        return []
    parts = re.split(r"[\n,]+", raw)
    return [p.strip() for p in parts if p.strip()]
