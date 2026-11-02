"""Honest evaluation for RESULTS.md. Run:
    python scripts/evaluate.py

(A) Classifier, GroupKFold by template_family, with two embeddings:
    TF-IDF + LogisticRegression (baseline) and MiniLM embeddings +
    LogisticRegression. Same folds for both (deterministic GroupKFold
    on the same row order), so the side-by-side comparison is fair.
(B) Retrieval-only kNN over Chroma (top-5 majority vote), same folds,
    with TF-IDF vectors and (when available) MiniLM vectors.
(C)/(D) LLM paths: only if LLM_API_KEY is set, else 'LLM not evaluated'.
    See scripts/llm_evaluate.py for the cached LLM evaluation.
Also: release-decision check on 10 handcrafted runs.

Prints machine-readable lines (EVAL_...) plus a confusion matrix per
method. Nothing here invents numbers: every figure comes from a run.
If MiniLM cannot be loaded, its lines print MINILM NOT AVAILABLE with
the error, and the TF-IDF numbers still stand.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from app.categories import ALL_CATEGORIES
from app.classifier import failure_text, load_labeled_rows
from app.rules import decide

# Imported lazily in main() where needed (they pull heavy deps).
LABELS = ALL_CATEGORIES


def minilm_encode(texts: list[str]) -> list[list[float]]:
    """Encode texts with all-MiniLM-L6-v2 (384-dim, L2-normalised).

    Raises if sentence-transformers or the model cannot be loaded; the
    caller turns that into a MINILM NOT AVAILABLE line, not a crash.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors, dtype=float).tolist()


def _score(y_true: list[str], y_pred: list[str]):
    return (accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
            confusion_matrix(y_true, y_pred, labels=LABELS).tolist())


def evaluate_sklearn(rows, groups, n_splits=5, embedding: str = "tfidf"):
    """(A) GroupKFold classifier. embedding='tfidf' uses the TF-IDF +
    LogisticRegression pipeline; embedding='minilm' uses MiniLM vectors
    + LogisticRegression. Returns (accuracy, macro_f1, confusion, y_true, y_pred)."""
    texts = [failure_text(r["test_name"], r["message"], r["log"]) for r in rows]
    labels = [r["category"] for r in rows]
    gkf = GroupKFold(n_splits=n_splits)
    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    all_vectors = minilm_encode(texts) if embedding == "minilm" else None
    for train_idx, test_idx in gkf.split(texts, labels, groups):
        if embedding == "minilm":
            assert all_vectors is not None
            clf = LogisticRegression(max_iter=1000)
            clf.fit([all_vectors[i] for i in train_idx], [labels[i] for i in train_idx])
            preds = [str(p) for p in clf.predict([all_vectors[i] for i in test_idx])]
        else:
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
                ("clf", LogisticRegression(max_iter=1000)),
            ])
            pipe.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
            preds = [str(p) for p in pipe.predict([texts[i] for i in test_idx])]
        y_true_all.extend(labels[i] for i in test_idx)
        y_pred_all.extend(preds)
    acc, f1, cm = _score(y_true_all, y_pred_all)
    return acc, f1, cm, y_true_all, y_pred_all


def evaluate_knn_chroma(rows, groups, n_splits=5, k=5, embedding: str = "tfidf"):
    """(B) Retrieval-only kNN over Chroma: per fold, index the train fold
    in an ephemeral Chroma collection, query each test item, majority vote
    of the top-k neighbours (ties -> nearest neighbour's category).
    embedding='tfidf' fits a TF-IDF vectorizer on the train fold only
    (no test leakage); embedding='minilm' uses pre-computed MiniLM
    vectors (encoding is not fitted, so no leakage either)."""
    import uuid

    import chromadb
    from chromadb.config import Settings

    texts = [failure_text(r["test_name"], r["message"], r["log"]) for r in rows]
    labels = [r["category"] for r in rows]
    gkf = GroupKFold(n_splits=n_splits)
    y_true_all: list[str] = []
    y_pred_all: list[str] = []
    all_vectors = minilm_encode(texts) if embedding == "minilm" else None
    for fold, (train_idx, test_idx) in enumerate(gkf.split(texts, labels, groups)):
        train_texts = [texts[i] for i in train_idx]
        if embedding == "minilm":
            assert all_vectors is not None
            train_vectors = [all_vectors[i] for i in train_idx]
            test_vectors = [all_vectors[i] for i in test_idx]
        else:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            train_vectors = vectorizer.fit_transform(train_texts).toarray().astype(float).tolist()
            test_vectors = vectorizer.transform([texts[i] for i in test_idx]).toarray().astype(float).tolist()
        client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
        # Unique name per call: this Chroma version shares segment state
        # across EphemeralClient instances, so a fixed name collides when
        # (B) runs twice in one process (TF-IDF, then MiniLM).
        collection = client.create_collection(
            name=f"eval_fold_{fold}_{uuid.uuid4().hex[:8]}",
            metadata={"hnsw:space": "cosine"},
        )
        collection.add(
            documents=train_texts,
            embeddings=train_vectors,
            metadatas=[{"category": labels[i]} for i in train_idx],
            ids=[f"train-{i}" for i in range(len(train_idx))],
        )
        result = collection.query(
            query_embeddings=test_vectors,
            n_results=min(k, len(train_idx)),
        )
        metadatas = result.get("metadatas") or []
        for j, test_i in enumerate(test_idx):
            neighbours = [m["category"] for m in (metadatas[j] if j < len(metadatas) else [])]
            if not neighbours:
                pred = labels[train_idx[0]]
            else:
                counts = Counter(neighbours)
                top = counts.most_common()
                best_count = top[0][1]
                winners = [c for c, n in top if n == best_count]
                pred = neighbours[0] if len(winners) > 1 else winners[0]
            y_true_all.append(labels[test_i])
            y_pred_all.append(pred)
    return (accuracy_score(y_true_all, y_pred_all),
            f1_score(y_true_all, y_pred_all, labels=LABELS, average="macro", zero_division=0),
            confusion_matrix(y_true_all, y_pred_all, labels=LABELS).tolist())


# 10 handcrafted release-decision runs: (name, pass_rate, failures, expected)
# failures: (category, critical). Expected verdicts follow rules.yaml exactly.
DECISION_RUNS = [
    ("all_pass", 100.0, [], "GREEN"),
    ("flaky_only_high_pass", 99.0, [("flaky", False)], "YELLOW"),
    ("env_only_high_pass", 98.0, [("environment", False)], "YELLOW"),
    ("pass_rate_95_mixed", 95.0, [("real_bug", False)], "YELLOW"),
    ("pass_rate_90_boundary", 90.0, [("test_data", False)], "YELLOW"),
    ("pass_rate_89", 89.0, [("flaky", False)], "RED"),
    ("critical_real_bug", 96.0, [("real_bug", True)], "RED"),
    ("critical_real_bug_high_pass", 99.0, [("real_bug", True)], "RED"),
    ("dependency_mid_pass", 93.0, [("dependency", False)], "YELLOW"),
    ("pass_rate_97_boundary", 97.0, [], "GREEN"),
]


def decision_check():
    agree = 0
    details = []
    for name, rate, failures, expected in DECISION_RUNS:
        verdict, rule = decide(rate, [{"category": c, "critical": crit} for c, crit in failures])
        ok = verdict == expected
        agree += int(ok)
        details.append({"name": name, "expected": expected, "got": verdict, "rule": rule, "agree": ok})
    return agree, details


def main() -> None:
    rows = load_labeled_rows()
    groups = [r["template_family"] for r in rows]
    print(f"Rows: {len(rows)}, families: {len(set(groups))}")
    acc, f1, cm, _, _ = evaluate_sklearn(rows, groups)
    print(f"EVAL_A_ACCURACY={acc:.4f}")
    print(f"EVAL_A_MACRO_F1={f1:.4f}")
    print("EVAL_A_CONFUSION_LABELS=" + json.dumps(LABELS))
    print("EVAL_A_CONFUSION=" + json.dumps(cm))
    acc_b, f1_b, cm_b = evaluate_knn_chroma(rows, groups)
    print(f"EVAL_B_ACCURACY={acc_b:.4f}")
    print(f"EVAL_B_MACRO_F1={f1_b:.4f}")
    print("EVAL_B_CONFUSION=" + json.dumps(cm_b))
    try:
        acc_am, f1_am, cm_am, _, _ = evaluate_sklearn(rows, groups, embedding="minilm")
        print(f"EVAL_A_MINILM_ACCURACY={acc_am:.4f}")
        print(f"EVAL_A_MINILM_MACRO_F1={f1_am:.4f}")
        print("EVAL_A_MINILM_CONFUSION=" + json.dumps(cm_am))
        acc_bm, f1_bm, cm_bm = evaluate_knn_chroma(rows, groups, embedding="minilm")
        print(f"EVAL_B_MINILM_ACCURACY={acc_bm:.4f}")
        print(f"EVAL_B_MINILM_MACRO_F1={f1_bm:.4f}")
        print("EVAL_B_MINILM_CONFUSION=" + json.dumps(cm_bm))
    except Exception as exc:  # noqa: BLE001 - MiniLM unavailability is a result
        print(f"MINILM NOT AVAILABLE: {type(exc).__name__}: {exc}")
    agree, details = decision_check()
    print(f"EVAL_DECISION_AGREEMENT={agree}/10")
    for d in details:
        print(f"EVAL_DECISION {d['name']} expected={d['expected']} got={d['got']} rule={d['rule']} agree={d['agree']}")


if __name__ == "__main__":
    main()
