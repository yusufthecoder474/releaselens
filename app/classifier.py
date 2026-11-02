"""Failure classifier: scikit-learn TF-IDF + LogisticRegression.

Trained on data/failures_labeled.csv (message + log text). The input text
for a failure is "test_name + message + log". predict() returns the
category and the model's confidence (max predicted probability).
"""
from __future__ import annotations

import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .categories import ALL_CATEGORIES
from .config import labeled_csv

_classifier: Pipeline | None = None


def failure_text(test_name: str, message: str, log: str) -> str:
    return f"{test_name} {message} {log}".strip()


def load_labeled_rows(csv_path: Path | None = None) -> list[dict[str, str]]:
    path = csv_path or labeled_csv()
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def train_classifier(csv_path: Path | None = None) -> Pipeline:
    rows = load_labeled_rows(csv_path)
    texts = [failure_text(r["test_name"], r["message"], r["log"]) for r in rows]
    labels = [r["category"] for r in rows]
    pipe: Pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    pipe.fit(texts, labels)
    return pipe


def get_classifier() -> Pipeline:
    """Lazily trained singleton (training on 180 rows takes <1s)."""
    global _classifier
    if _classifier is None:
        _classifier = train_classifier()
    return _classifier


def predict_category(test_name: str, message: str, log: str) -> tuple[str, float]:
    """Return (category, confidence). Confidence is max class probability."""
    clf = get_classifier()
    text = failure_text(test_name, message, log)
    proba = clf.predict_proba([text])[0]
    classes = list(clf.classes_)
    best = int(proba.argmax())
    category = str(classes[best])
    if category not in ALL_CATEGORIES:  # defensive; training data only has these
        category = "flaky"
    return category, float(proba[best])
