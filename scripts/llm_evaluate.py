"""LLM evaluation for ReleaseLens (task C, D, and prompt v1 vs v2).

(C) LLM + retrieval: for each GroupKFold fold, the LLM classifies the
    fold's TEST items using retrieved similar failures from the fold's
    TRAIN items only (Chroma, top-3). No self-leakage: test items are
    never in the retrieval memory for their own fold.
(D) Full agent end to end: a JUnit XML built from a fixed 20-failure
    subset is run through run_pipeline; final categories are compared
    with true labels.
Prompt v1 vs v2: (C) is run once per prompt version on the same folds.

Outputs are cached to disk (eval_cache/llm_cache.json) keyed by method,
prompt version, model and item, so reruns are cheap and reproducible.
Without LLM_API_KEY, the real runs cannot execute: main() prints
"LLM NOT EVALUATED" and exits without inventing numbers. The cache
loader and metric functions are still unit-tested with a fake LLM
(see tests/test_llm_eval.py).

Metrics per method: accuracy, macro-F1, confusion matrix, grounding
rate (RAW evidence lines found verbatim in the log, before the verify
guard; post-guard grounding is 1.0 by construction), invalid-category
rate BEFORE the verify guard, and (for D) tool-error rate.
Also recorded: model name, number of LLM calls, total time.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.config import Settings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from app.agent.graph import llm_explanation, run_pipeline, verify_explanations
from app.categories import ALL_CATEGORIES, LABELS
from app.classifier import failure_text, load_labeled_rows
from app.config import llm_api_key, llm_model, prompts_dir

LABELS = ALL_CATEGORIES
CACHE_PATH = Path(__file__).resolve().parents[1] / "eval_cache" / "llm_cache.json"


def load_cache(path: Path = CACHE_PATH) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict, path: Path = CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")


def grounding_rate_raw(evidence_lines: list[str], log: str) -> tuple[int, int]:
    """(kept, total) for raw LLM evidence lines against the log."""
    kept = sum(1 for ln in evidence_lines if ln.strip() and ln.strip() in log)
    return kept, len(evidence_lines)


def fold_splits(rows, groups, n_splits: int = 5):
    texts = [failure_text(r["test_name"], r["message"], r["log"]) for r in rows]
    labels = [r["category"] for r in rows]
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(texts, labels, groups))


def build_fold_collection(train_rows, train_texts):
    """Ephemeral Chroma collection over a fold's train items (TF-IDF
    vectors, cosine). Used for retrieval in (C). The name is unique per
    call because this Chroma version shares segment state across
    EphemeralClient instances, so a fixed name collides on later folds."""
    import uuid

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    vectors = vectorizer.fit_transform(train_texts).toarray().astype(float).tolist()
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    collection = client.create_collection(name=f"llm_eval_fold_{uuid.uuid4().hex[:8]}",
                                          metadata={"hnsw:space": "cosine"})
    collection.add(documents=train_texts, embeddings=vectors,
                   metadatas=[{"test_name": r["test_name"], "category": r["category"],
                               "message": r["message"]} for r in train_rows],
                   ids=[f"train-{i}" for i in range(len(train_rows))])
    return collection, vectorizer


def similar_from_collection(collection, vectorizer, query_text: str, k: int = 3) -> list[dict]:
    vec = vectorizer.transform([query_text]).toarray().astype(float).tolist()
    result = collection.query(query_embeddings=vec, n_results=min(k, collection.count()))
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    return [{"test_name": (m or {}).get("test_name", ""),
             "category": (m or {}).get("category", ""),
             "message": (m or {}).get("message", ""),
             "distance": float(d) if d is not None else 0.0}
            for m, d in zip(metadatas, distances)]


def run_c(rows, llm_client, prompt_version: str, model: str,
          cache: dict, cache_path: Path) -> dict:
    """(C) LLM + retrieval over all folds. Returns metrics + counters."""
    from app.classifier import predict_category
    from app.prompts import load_prompt

    groups = [r["template_family"] for r in rows]
    texts = [failure_text(r["test_name"], r["message"], r["log"]) for r in rows]
    labels = [r["category"] for r in rows]
    prompt_template = load_prompt(prompt_version)
    y_true: list[str] = []
    y_pred: list[str] = []
    raw_kept = raw_total = 0
    invalid = 0
    llm_calls = 0
    llm_errors = 0
    start = time.perf_counter()
    for train_idx, test_idx in fold_splits(rows, groups):
        train_rows = [rows[i] for i in train_idx]
        train_texts = [texts[i] for i in train_idx]
        collection, vectorizer = build_fold_collection(train_rows, train_texts)
        for test_i in test_idx:
            row = rows[test_i]
            key = f"C:{prompt_version}:{model}:{row['template_family']}:{row['test_name']}"
            # The prompt gets the CLASSIFIER's category, never the
            # true label (using the label would leak the answer).
            clf_cat, _ = predict_category(row["test_name"], row["message"], row["log"])
            if key in cache:
                raw = cache[key]
                # A cached error placeholder still represents a failed
                # LLM call; keep it visible in the error count on reruns.
                if raw.get("category") == "__llm_error__":
                    llm_errors += 1
            else:
                similar = similar_from_collection(collection, vectorizer, texts[test_i])
                failure = {"test_name": row["test_name"], "message": row["message"],
                           "log": row["log"]}
                raw = llm_explanation(llm_client, prompt_template, failure,
                                      clf_cat, similar)
                llm_calls += 1
                if raw is None:
                    # LLM call failed (llm_explanation returns None on
                    # any error). Recorded as an LLM error, not as an
                    # invalid category; the guard fallback below still
                    # gives the system an output category.
                    llm_errors += 1
                    raw = {"test_name": row["test_name"], "category": "__llm_error__",
                           "plain_english_reason": "", "evidence_lines": [],
                           "suggested_next_step": "", "nearest": None}
                cache[key] = raw
                save_cache(cache, cache_path)
            y_true.append(row["category"])
            raw_category = raw.get("category", "")
            # Invalid-category rate counts only real LLM outputs with a
            # category outside the enum (not errors, not cache hits of
            # an error placeholder — those are counted as llm_errors
            # when first produced, and cache hits replay them).
            if raw_category not in ALL_CATEGORIES and raw_category != "__llm_error__":
                invalid += 1
            # Guard-verified category is the system's actual output.
            verified, _, _, _ = verify_explanations(
                [raw], {row["test_name"]: {"log": row["log"],
                                           "classifier_category": clf_cat}})
            y_pred.append(verified[0]["category"])
            kept, total = grounding_rate_raw(raw.get("evidence_lines", []), row["log"])
            raw_kept += kept
            raw_total += total
    elapsed = time.perf_counter() - start
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "grounding_rate_raw": (raw_kept / raw_total) if raw_total else 1.0,
        "grounding_lines": f"{raw_kept}/{raw_total}",
        "invalid_category_rate": invalid / len(y_true) if y_true else 0.0,
        "invalid_count": invalid,
        "llm_calls": llm_calls,
        "llm_errors": llm_errors,
        "total_time_s": round(elapsed, 3),
        "n_items": len(y_true),
    }


def build_demo_xml(rows) -> str:
    """Build a JUnit XML document from fixed rows (for (D))."""
    import xml.etree.ElementTree as ET

    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", {"name": "llm_eval_demo",
                                              "tests": str(len(rows)),
                                              "failures": str(len(rows))})
    for r in rows:
        tc = ET.SubElement(suite, "testcase", {"classname": "T", "name": r["test_name"]})
        fail = ET.SubElement(tc, "failure", {"message": r["message"]})
        fail.text = r["log"]
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def run_d(rows, llm_client) -> dict:
    """(D) Full agent end to end on 20 fixed failures (4 per category,
    deterministic: first 4 rows per category in file order)."""
    from app.agent.graph import run_pipeline

    subset: list[dict] = []
    by_cat: dict[str, list[dict]] = {c: [] for c in ALL_CATEGORIES}
    for r in rows:
        if len(by_cat[r["category"]]) < 4:
            by_cat[r["category"]].append(r)
    for c in ALL_CATEGORIES:
        subset.extend(by_cat[c])
    xml_text = build_demo_xml(subset)
    start = time.perf_counter()
    report = run_pipeline(xml_text, source="llm_eval_demo", llm_client=llm_client)
    elapsed = time.perf_counter() - start
    pred_by_test: dict[str, str] = {}
    for group in report["groups"]:
        for entry in group["entries"]:
            pred_by_test[entry["test_name"]] = group["category"]
    y_true = [r["category"] for r in subset]
    y_pred = [pred_by_test.get(r["test_name"], "__missing__") for r in subset]
    notes = report.get("investigation_notes", [])
    tool_outputs = [n for n in notes if not n.startswith("LLM conclusion:")]
    tool_errors = sum(1 for n in tool_outputs if "failed:" in n)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "grounding_rate": report.get("grounding_rate", 0.0),
        "invalid_category_count": report.get("invalid_category_count", 0),
        "tool_error_rate": (tool_errors / len(tool_outputs)) if tool_outputs else 0.0,
        "tool_errors": tool_errors,
        "tool_calls": len(tool_outputs),
        "total_time_s": round(elapsed, 3),
        "n_items": len(subset),
        "verdict": report.get("verdict"),
    }


def main() -> None:
    if not llm_api_key():
        print("LLM NOT EVALUATED: LLM_API_KEY is not set in this session.")
        print("No LLM numbers are claimed. See RESULTS.md for the exact blocker.")
        return
    from langchain_openai import ChatOpenAI

    from app.config import llm_base_url

    model = llm_model()
    llm_client = ChatOpenAI(model=model, base_url=llm_base_url(),
                            api_key=llm_api_key(), temperature=0, timeout=120)
    rows = load_labeled_rows()
    cache = load_cache()
    print(f"Model: {model}")
    for version in ("v1", "v2"):
        metrics = run_c(rows, llm_client, version, model, cache, CACHE_PATH)
        print(f"EVAL_C_{version.upper()} " + json.dumps(metrics))
    metrics_d = run_d(rows, llm_client)
    print("EVAL_D " + json.dumps(metrics_d))


if __name__ == "__main__":
    main()
