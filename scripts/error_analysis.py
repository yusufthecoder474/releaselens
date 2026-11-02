"""Error analysis for the Dependency category + one tried change.

Runs the GroupKFold (A) baseline, lists Dependency confusions with the
actual failure messages, then tries ONE change — adding character
n-gram features (char_wb 3-5) alongside word n-grams — and reports
before/after numbers on the SAME folds. The change is kept only if it
helps; see RESULTS.md for the verdict.

Prints ERROR_ANALYSIS_... lines. Nothing is invented: every message
shown comes from the labeled data.
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

from app.categories import ALL_CATEGORIES
from app.classifier import failure_text, load_labeled_rows

LABELS = ALL_CATEGORIES


def run_groupkfold(rows, groups, make_pipe):
    texts = [failure_text(r["test_name"], r["message"], r["log"]) for r in rows]
    labels = [r["category"] for r in rows]
    gkf = GroupKFold(n_splits=5)
    y_true: list[str] = []
    y_pred: list[str] = []
    items: list[dict] = []
    for train_idx, test_idx in gkf.split(texts, labels, groups):
        pipe = make_pipe()
        pipe.fit([texts[i] for i in train_idx], [labels[i] for i in train_idx])
        preds = [str(p) for p in pipe.predict([texts[i] for i in test_idx])]
        for j, test_i in enumerate(test_idx):
            y_true.append(labels[test_i])
            y_pred.append(preds[j])
            items.append({"row": rows[test_i], "true": labels[test_i], "pred": preds[j]})
    return y_true, y_pred, items


def baseline_pipe():
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def char_augmented_pipe():
    """ONE tried change: word n-grams + character n-grams."""
    return Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
        ])),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def main() -> None:
    rows = load_labeled_rows()
    groups = [r["template_family"] for r in rows]
    y_true, y_pred, items = run_groupkfold(rows, groups, baseline_pipe)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    print(f"ERROR_BASELINE_ACCURACY={acc:.4f} MACRO_F1={f1:.4f}")
    dep_errors = [it for it in items if it["true"] == "dependency" and it["pred"] != "dependency"]
    print(f"ERROR_DEPENDENCY_ERRORS={len(dep_errors)} of 36")
    by_pred: dict[str, list[dict]] = defaultdict(list)
    for it in dep_errors:
        by_pred[it["pred"]].append(it)
    for pred, lst in sorted(by_pred.items(), key=lambda kv: -len(kv[1])):
        print(f"ERROR_CONFUSION dependency -> {pred}: {len(lst)}")
        for it in lst[:3]:
            r = it["row"]
            print(f"  - [{r['template_family']}] {r['test_name']}: {r['message']}")
    y_true2, y_pred2, _ = run_groupkfold(rows, groups, char_augmented_pipe)
    acc2 = accuracy_score(y_true2, y_pred2)
    f1_2 = f1_score(y_true2, y_pred2, labels=LABELS, average="macro", zero_division=0)
    dep_correct2 = sum(1 for t, p in zip(y_true2, y_pred2)
                       if t == "dependency" and p == "dependency")
    print(f"ERROR_CHAR_AUGMENTED_ACCURACY={acc2:.4f} MACRO_F1={f1_2:.4f} "
          f"DEPENDENCY_CORRECT={dep_correct2}/36")
    print(f"ERROR_VERDICT={'KEEP char n-grams' if (f1_2 > f1 or acc2 > acc) else 'DISCARD char n-grams'}")


if __name__ == "__main__":
    main()
