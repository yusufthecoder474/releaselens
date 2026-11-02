"""LangGraph state for one release-readiness run."""
from __future__ import annotations

from typing import TypedDict


class AgentState(TypedDict, total=False):
    xml_text: str
    source: str
    # Parsed run: totals, pass_rate, and test dicts.
    parsed: dict
    failures: list[dict]
    classifications: list[dict]
    routed_to: str  # "explain" | "investigate"
    investigation_notes: list[str]
    explanations: list[dict]
    verified: list[dict]
    invalid_category_count: int
    unverified_evidence_count: int
    grounding_rate: float
    verdict: str
    rule_key: str
    rule_description: str
    confidence_label: str
    recheck: list[dict]
    flaky: list[str]
    llm_used: bool
    prompt_version: str
    report: dict
