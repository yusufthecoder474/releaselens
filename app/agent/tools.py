"""LangChain tools for the investigate node.

Each tool is a real LangChain Tool. Tool errors are never raised to the
caller: safe_call() catches them and returns the error as text, so a
failing tool cannot crash the agent (see tests/test_tools.py).
"""
from __future__ import annotations

from langchain_core.tools import Tool

from ..classifier import failure_text
from ..flaky import test_history
from ..memory import search_similar


class ToolContext:
    """What tools need: the Chroma collection, a DB session, and the
    current run's failures (test_name -> log text)."""

    def __init__(self, collection, session, failures: dict[str, str]) -> None:
        self.collection = collection
        self.session = session
        self.failures = failures


def safe_call(tool: Tool, arg: str) -> str:
    """Invoke a tool, returning errors as text instead of raising."""
    try:
        result = tool.invoke(arg)
        return str(result)
    except Exception as exc:  # noqa: BLE001 - tool errors are data here
        return f"Tool {tool.name} failed: {exc}"


def build_tools(ctx: ToolContext) -> dict[str, Tool]:
    """Build the three investigate tools bound to this run's context."""

    def _search(query: str) -> str:
        similar = search_similar(ctx.collection, query, k=3)
        if not similar:
            return "No similar past failures found."
        lines = [f"{s['test_name']} ({s['category']}, distance {s['distance']:.3f})"
                 for s in similar]
        return "Similar past failures:\n" + "\n".join(lines)

    def _history(test_name: str) -> str:
        rows = test_history(ctx.session, test_name.strip())
        if not rows:
            return f"No history found for test '{test_name}'."
        lines = [f"run {r['run_id']}: {r['status']} ({r['created_at']})" for r in rows]
        return f"History for '{test_name}':\n" + "\n".join(lines)

    def _excerpt(test_name: str) -> str:
        log = ctx.failures.get(test_name.strip())
        if log is None:
            return f"No log found for test '{test_name}' in this run."
        return log[:500] if log else "(empty log)"

    return {
        "search_similar_failures": Tool(
            name="search_similar_failures",
            description="Find similar past failures for a failure message/log text. Input: the failure text.",
            func=_search,
        ),
        "get_test_history": Tool(
            name="get_test_history",
            description="Get the recent run history (passed/failed) for one test name. Input: the test name.",
            func=_history,
        ),
        "get_log_excerpt": Tool(
            name="get_log_excerpt",
            description="Get the first 500 characters of the log for one test in this run. Input: the test name.",
            func=_excerpt,
        ),
    }


def tool_descriptions(tools: dict[str, Tool]) -> str:
    return "\n".join(f"- {t.name}: {t.description}" for t in tools.values())


def failure_text_of(failure: dict) -> str:
    return failure_text(failure["test_name"], failure.get("message", ""), failure.get("log", ""))
