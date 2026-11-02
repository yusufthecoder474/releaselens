"""LangGraph agent: parse -> classify -> route -> investigate -> explain
-> verify -> decide -> render.

Design principle: the LLM only EXPLAINS. The verdict comes from the
deterministic rules in rules.yaml (decide node). In no-key mode the
explain node uses a deterministic template, so the whole pipeline works
with no LLM key at all.

Routing is per-run: if the lowest classifier confidence across failures
is >= 0.75, the graph goes straight to explain; otherwise it goes to
investigate first (max 3 tool calls, errors returned as text).
"""
from __future__ import annotations

import re

from langgraph.graph import END, START, StateGraph

from ..categories import ALL_CATEGORIES, LABELS, NEXT_STEPS
from ..classifier import failure_text, predict_category
from ..config import llm_api_key, llm_base_url, llm_model, prompt_version
from ..db import Failure, Run, TestResult, get_session, init_db
from ..flaky import flaky_tests, recheck_first
from ..memory import get_collection, search_similar
from ..parser import parse_junit_xml
from ..prompts import load_prompt
from ..rules import decide, is_critical, load_rules, rule_description
from ..schemas import Explanation
from ..seed import seed_historical_runs
from . import tools as tools_mod
from .state import AgentState
from .tools import ToolContext, failure_text_of, safe_call, tool_descriptions

CONFIDENCE_THRESHOLD = 0.75
MAX_TOOL_CALLS = 3

WHAT_IT_MEANS = {
    "GREEN": "This release looks ready: the tests pass well enough and no blocking failure was found.",
    "YELLOW": "This release needs a second look: the failures are only flaky or environment problems, or the pass rate is below 97%.",
    "RED": "Do not release yet: a blocking failure or a low pass rate was found.",
}


def template_explanation(failure: dict, category: str, similar: list[dict]) -> dict:
    """Deterministic no-key explanation built from the classifier result
    and the nearest past failure. Evidence lines are copied verbatim
    from the log, so grounding is exact by construction."""
    label = LABELS[category]
    reason = f"This looks like {label.lower()}: {failure.get('message', '')}".strip()
    if similar:
        nearest = similar[0]
        reason += (f" A similar past failure was '{nearest['test_name']}' "
                   f"({LABELS.get(nearest['category'], nearest['category'])}).")
    log_lines = [ln.strip() for ln in failure.get("log", "").splitlines() if ln.strip()]
    return {
        "test_name": failure["test_name"],
        "category": category,
        "plain_english_reason": reason,
        "evidence_lines": log_lines[:3],
        "suggested_next_step": NEXT_STEPS[category],
        "nearest": ({"test_name": similar[0]["test_name"], "category": similar[0]["category"]}
                    if similar else None),
    }


def llm_explanation(llm, prompt_template: str, failure: dict, category: str,
                    similar: list[dict]) -> dict | None:
    """LLM explanation with Pydantic structured output. Returns None on
    any failure so the caller can fall back to the template."""
    similar_text = "none"
    if similar:
        similar_text = "\n".join(
            f"- {s['test_name']} ({s['category']}): {s.get('message', '')}" for s in similar)
    prompt = prompt_template.format(
        test_name=failure["test_name"], category=category,
        message=failure.get("message", ""), log=failure.get("log", ""),
        similar=similar_text)
    try:
        structured = llm.with_structured_output(Explanation)
        result: Explanation = structured.invoke(prompt)
        return {
            "test_name": failure["test_name"],
            "category": result.category.value,
            "plain_english_reason": result.plain_english_reason,
            "evidence_lines": list(result.evidence_lines),
            "suggested_next_step": result.suggested_next_step,
            "nearest": ({"test_name": similar[0]["test_name"], "category": similar[0]["category"]}
                        if similar else None),
        }
    except Exception:
        return None


def verify_explanations(explanations: list[dict], failures_by_name: dict
                        ) -> tuple[list[dict], int, int, float]:
    """Verify guard (pure function, unit-tested directly).

    - Category not in the enum -> fall back to the classifier's category.
    - Evidence line not verbatim in the original log -> dropped, counted
      as unverified.
    Returns (verified, invalid_category_count, unverified_count, grounding_rate).
    """
    verified: list[dict] = []
    invalid = 0
    unverified = 0
    total_lines = 0
    kept_lines = 0
    for exp in explanations:
        name = exp["test_name"]
        failure = failures_by_name.get(name, {})
        log = failure.get("log", "")
        classifier_category = failure.get("classifier_category", "flaky")
        category = exp.get("category", "")
        if category not in ALL_CATEGORIES:
            category = classifier_category
            invalid += 1
        kept: list[str] = []
        for line in exp.get("evidence_lines", []):
            total_lines += 1
            stripped = line.strip()
            if stripped and stripped in log:
                kept.append(line)
                kept_lines += 1
            else:
                unverified += 1
        verified.append({
            "test_name": name,
            "category": category,
            "plain_english_reason": exp.get("plain_english_reason", ""),
            "evidence_lines": kept,
            "suggested_next_step": exp.get("suggested_next_step", ""),
            "nearest": exp.get("nearest"),
        })
    grounding = (kept_lines / total_lines) if total_lines else 1.0
    return verified, invalid, unverified, grounding


class Ctx:
    """Per-run context the graph nodes close over (not stored in state)."""

    def __init__(self, collection, session, rules: dict, prompt_text: str,
                 prompt_ver: str, llm) -> None:
        self.collection = collection
        self.session = session
        self.rules = rules
        self.prompt_text = prompt_text
        self.prompt_version = prompt_ver
        self.llm = llm


def build_graph(ctx: Ctx):
    """Build the StateGraph. Nodes are closures over ctx."""

    def parse_node(state: AgentState) -> dict:
        parsed = parse_junit_xml(state["xml_text"])
        tests = [{
            "test_name": t.name, "classname": t.classname, "status": t.status,
            "message": t.message, "log": t.log,
            "critical_property": t.critical_property,
        } for t in parsed.tests]
        failures = [t for t in tests if t["status"] == "failed"]
        return {
            "parsed": {"total": parsed.total, "passed": parsed.passed,
                       "failed": parsed.failed, "skipped": parsed.skipped,
                       "pass_rate": parsed.pass_rate, "tests": tests},
            "failures": failures,
        }

    def classify_node(state: AgentState) -> dict:
        out = []
        for f in state["failures"]:
            category, confidence = predict_category(f["test_name"], f.get("message", ""), f.get("log", ""))
            out.append({"test_name": f["test_name"], "category": category,
                        "confidence": confidence,
                        "critical_property": f.get("critical_property", False)})
        return {"classifications": out}

    def route(state: AgentState) -> str:
        classifications = state.get("classifications", [])
        if not classifications:
            return "explain"
        min_conf = min(c["confidence"] for c in classifications)
        return "explain" if min_conf >= CONFIDENCE_THRESHOLD else "investigate"

    def investigate_node(state: AgentState) -> dict:
        classifications = state.get("classifications", [])
        if not classifications:
            return {"routed_to": "investigate", "investigation_notes": []}
        target = min(classifications, key=lambda c: c["confidence"])
        failure = next((f for f in state["failures"] if f["test_name"] == target["test_name"]),
                       state["failures"][0])
        failures_map = {f["test_name"]: f.get("log", "") for f in state["failures"]}
        tool_ctx = ToolContext(ctx.collection, ctx.session, failures_map)
        tools = tools_mod.build_tools(tool_ctx)
        notes: list[str] = []
        if ctx.llm is not None:
            notes = _llm_guided_investigation(ctx.llm, tools, failure)
        if not notes:
            # No-key mode, or the LLM loop produced nothing: direct calls.
            calls = [
                ("search_similar_failures", failure_text_of(failure)),
                ("get_test_history", failure["test_name"]),
                ("get_log_excerpt", failure["test_name"]),
            ]
            for name, arg in calls[:MAX_TOOL_CALLS]:
                notes.append(f"{name}: {safe_call(tools[name], arg)}")
        return {"routed_to": "investigate", "investigation_notes": notes[:MAX_TOOL_CALLS]}

    def explain_node(state: AgentState) -> dict:
        routed = state.get("routed_to", "explain")
        # classify -> route sets routed_to only on the investigate path;
        # the direct path also lands here, so record it if unset.
        if "routed_to" not in state:
            routed = "explain"
        failures_by_name = {f["test_name"]: f for f in state["failures"]}
        explanations = []
        llm_used = False
        for c in state["classifications"]:
            failure = failures_by_name[c["test_name"]]
            similar = search_similar(ctx.collection, failure_text_of(failure), k=3)
            exp = None
            if ctx.llm is not None:
                exp = llm_explanation(ctx.llm, ctx.prompt_text, failure,
                                      c["category"], similar)
                if exp is not None:
                    llm_used = True
            if exp is None:
                exp = template_explanation(failure, c["category"], similar)
            explanations.append(exp)
        return {"routed_to": routed, "explanations": explanations, "llm_used": llm_used}

    def verify_node(state: AgentState) -> dict:
        failures_by_name = {}
        for c in state["classifications"]:
            failure = next((f for f in state["failures"] if f["test_name"] == c["test_name"]), {})
            failures_by_name[c["test_name"]] = {
                "log": failure.get("log", ""), "classifier_category": c["category"]}
        verified, invalid, unverified, grounding = verify_explanations(
            state.get("explanations", []), failures_by_name)
        return {"verified": verified, "invalid_category_count": invalid,
                "unverified_evidence_count": unverified, "grounding_rate": grounding}

    def decide_node(state: AgentState) -> dict:
        parsed = state["parsed"]
        failures_for_rules = []
        for v in state.get("verified", []):
            c = next((x for x in state["classifications"] if x["test_name"] == v["test_name"]), {})
            failures_for_rules.append({
                "category": v["category"],
                "critical": is_critical(v["test_name"], c.get("critical_property", False), ctx.rules),
            })
        verdict, rule_key = decide(parsed["pass_rate"], failures_for_rules, ctx.rules)
        confs = [c["confidence"] for c in state["classifications"]]
        mean_conf = sum(confs) / len(confs) if confs else 1.0
        label = "High" if mean_conf >= 0.75 else ("Medium" if mean_conf >= 0.5 else "Low")
        return {"verdict": verdict, "rule_key": rule_key,
                "rule_description": rule_description(rule_key),
                "confidence_label": label}

    def render_node(state: AgentState) -> dict:
        parsed = state["parsed"]
        session = ctx.session
        run = Run(source=state.get("source", "upload"), total=parsed["total"],
                  passed=parsed["passed"], failed=parsed["failed"],
                  verdict=state["verdict"], pass_rate=parsed["pass_rate"])
        session.add(run)
        session.flush()
        result_ids: dict[str, int] = {}
        for t in parsed["tests"]:
            tr = TestResult(run_id=run.id, test_name=t["test_name"], status=t["status"])
            session.add(tr)
            session.flush()
            result_ids[t["test_name"]] = tr.id
        conf_by_name = {c["test_name"]: c["confidence"] for c in state["classifications"]}
        for v in state.get("verified", []):
            failure = next((f for f in state["failures"] if f["test_name"] == v["test_name"]), {})
            session.add(Failure(test_result_id=result_ids.get(v["test_name"], 0),
                                test_name=v["test_name"], category=v["category"],
                                confidence=conf_by_name.get(v["test_name"], 0.0),
                                message=failure.get("message", ""),
                                log_text=failure.get("log", "")))
        session.commit()
        # History queries run after persisting, so this run counts too.
        recheck = recheck_first(session)
        flaky = flaky_tests(session)
        groups: dict[str, dict] = {}
        for v in state.get("verified", []):
            label = LABELS[v["category"]]
            groups.setdefault(v["category"], {"category": v["category"], "label": label,
                                              "count": 0, "entries": []})
            groups[v["category"]]["count"] += 1
            groups[v["category"]]["entries"].append({
                "test_name": v["test_name"],
                "plain_english_reason": v["plain_english_reason"],
                "evidence_lines": v["evidence_lines"],
                "suggested_next_step": v["suggested_next_step"],
                "nearest": v.get("nearest"),
            })
        report = {
            "run_id": run.id,
            "source": state.get("source", "upload"),
            "verdict": state["verdict"],
            "rule_key": state["rule_key"],
            "rule_description": state["rule_description"],
            "what_it_means": WHAT_IT_MEANS[state["verdict"]],
            "total": parsed["total"], "passed": parsed["passed"],
            "failed": parsed["failed"], "skipped": parsed["skipped"],
            "pass_rate": round(parsed["pass_rate"], 2),
            "groups": list(groups.values()),
            "recheck_first": recheck,
            "flaky_tests": flaky,
            "confidence_label": state["confidence_label"],
            "routed_to": state.get("routed_to", "explain"),
            "prompt_version": ctx.prompt_version,
            "llm_used": state.get("llm_used", False),
            "grounding_rate": round(state.get("grounding_rate", 1.0), 4),
            "unverified_evidence_count": state.get("unverified_evidence_count", 0),
            "invalid_category_count": state.get("invalid_category_count", 0),
            "investigation_notes": state.get("investigation_notes", []),
        }
        return {"recheck": recheck, "flaky": flaky, "report": report}

    graph = StateGraph(AgentState)
    graph.add_node("parse", parse_node)
    graph.add_node("classify", classify_node)
    graph.add_node("investigate", investigate_node)
    graph.add_node("explain", explain_node)
    graph.add_node("verify", verify_node)
    graph.add_node("decide", decide_node)
    graph.add_node("render", render_node)
    graph.add_edge(START, "parse")
    graph.add_edge("parse", "classify")
    graph.add_conditional_edges("classify", route, {"explain": "explain", "investigate": "investigate"})
    graph.add_edge("investigate", "explain")
    graph.add_edge("explain", "verify")
    graph.add_edge("verify", "decide")
    graph.add_edge("decide", "render")
    graph.add_edge("render", END)
    return graph.compile()


def _llm_guided_investigation(llm, tools: dict, failure: dict) -> list[str]:
    """Tiny ReAct loop (max 3 tool calls). Returns [] if the LLM cannot
    be used, so the caller falls back to direct tool calls."""
    notes: list[str] = []
    history = ""
    action_re = re.compile(r"Action:\s*([a-z_]+)", re.IGNORECASE)
    input_re = re.compile(r"Action Input:\s*(.+)", re.IGNORECASE | re.DOTALL)
    for _ in range(MAX_TOOL_CALLS):
        prompt = (
            "You are investigating a test failure. You may call one tool, or finish.\n"
            f"Tools:\n{tool_descriptions(tools)}\n"
            f"Failure: {failure['test_name']}: {failure.get('message', '')}\n"
            f"So far:\n{history or '(nothing yet)'}\n"
            "Reply with exactly one of:\n"
            "Action: <tool name>\nAction Input: <input>\n"
            "or:\nFinal answer: <one sentence>"
        )
        try:
            text = str(llm.invoke(prompt).content)
        except Exception:
            return notes
        if "final answer" in text.lower():
            notes.append(f"LLM conclusion: {text.strip()[:500]}")
            return notes
        m_action = action_re.search(text)
        m_input = input_re.search(text)
        if not m_action or m_action.group(1) not in tools:
            return notes
        arg = m_input.group(1).strip() if m_input else failure["test_name"]
        obs = safe_call(tools[m_action.group(1)], arg)
        notes.append(f"{m_action.group(1)}: {obs}")
        history += f"\n- {m_action.group(1)}({arg[:80]}): {obs[:300]}"
    return notes


def _make_llm():
    """ChatOpenAI for an OpenAI-compatible endpoint, or None (no key)."""
    key = llm_api_key()
    if not key:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=llm_model(), base_url=llm_base_url(),
                      api_key=key, temperature=0)


def run_pipeline(xml_text: str, source: str = "upload", session=None,
                 llm_client=None) -> dict:
    """Run the full agent pipeline on one JUnit XML document.

    Returns the report dict. Works with no LLM key (template explain).
    llm_client, when given, is used instead of building one from the
    environment (this is the seam the LLM evaluation and its fake-LLM
    tests use); pass None (default) for the normal env-based client.
    """
    own_session = session is None
    if session is None:
        session = get_session()
    try:
        init_db()
        seed_historical_runs(session)
        _client, collection, _emb = get_collection()
        rules = load_rules()
        prompt_ver = prompt_version()
        llm = llm_client if llm_client is not None else _make_llm()
        ctx = Ctx(collection, session, rules, load_prompt(prompt_ver),
                  prompt_ver, llm)
        graph = build_graph(ctx)
        result = graph.invoke({"xml_text": xml_text, "source": source})
        return result["report"]
    finally:
        if own_session:
            session.close()
