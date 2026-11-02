"""Full pipeline works in no-key mode (no LLM key set by conftest)."""
import os

from app.agent.graph import run_pipeline
from app.config import samples_dir


def test_full_pipeline_no_key_real_bug_is_red():
    assert not os.environ.get("LLM_API_KEY")
    xml_text = (samples_dir() / "real_bug.xml").read_text(encoding="utf-8")
    report = run_pipeline(xml_text, source="test")
    assert report["verdict"] == "RED"
    assert report["rule_key"] == "critical_real_bug"
    assert report["llm_used"] is False
    assert report["groups"]
    labels = {g["label"] for g in report["groups"]}
    assert "Real product bug" in labels
    assert report["recheck_first"]


def test_full_pipeline_no_key_healthy_is_green():
    xml_text = (samples_dir() / "healthy.xml").read_text(encoding="utf-8")
    report = run_pipeline(xml_text, source="test")
    assert report["verdict"] == "GREEN"
    assert report["failed"] == 0
    assert report["groups"] == []
