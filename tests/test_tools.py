"""Tool-error handling: a failing tool must return error text, never
crash the agent — at the safe_call level and inside a full pipeline run.
"""
from langchain_core.tools import Tool

from app.agent import graph as graph_mod
from app.agent import tools as tools_mod
from app.agent.graph import run_pipeline
from app.agent.tools import safe_call
from app.config import samples_dir


def test_safe_call_returns_error_text_instead_of_raising():
    def boom(_: str) -> str:
        raise RuntimeError("boom")
    tool = Tool(name="broken", description="always fails", func=boom)
    result = safe_call(tool, "anything")
    assert "failed" in result.lower()
    assert "boom" in result


def test_pipeline_survives_failing_tool(monkeypatch):
    # Force the investigate route (low confidence) and break one tool.
    monkeypatch.setattr(graph_mod, "predict_category",
                        lambda name, msg, log: ("flaky", 0.5))
    # Capture the real builder BEFORE patching (else it would recurse).
    orig_build = tools_mod.build_tools

    def failing_build(ctx):
        tools = orig_build(ctx)

        def boom(_: str) -> str:
            raise RuntimeError("boom")
        tools["search_similar_failures"] = Tool(
            name="search_similar_failures", description="broken", func=boom)
        return tools

    # Patch the reference the graph actually calls.
    monkeypatch.setattr(tools_mod, "build_tools", failing_build)
    xml_text = (samples_dir() / "flaky_heavy.xml").read_text(encoding="utf-8")
    report = run_pipeline(xml_text, source="test")
    assert report["verdict"] in ("GREEN", "YELLOW", "RED")
    assert report["routed_to"] == "investigate"
    assert any("boom" in note for note in report["investigation_notes"])
