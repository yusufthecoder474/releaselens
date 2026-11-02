"""Vector memory: ChromaDB stores labeled past failures so new failures
can be compared with similar ones ("seen before").

Collection name includes the embedding name (past_failures_tfidf or
past_failures_minilm) so the two embedding choices never share a
collection. Seeding is idempotent: if the collection already has
documents, nothing is added again.
"""
from __future__ import annotations

from pathlib import Path

import chromadb

from .classifier import failure_text
from .config import chroma_path, labeled_csv
from .embeddings import get_embedding_function


def collection_name(embedding_name: str) -> str:
    return f"past_failures_{embedding_name}"


def get_collection(chroma_dir: Path | None = None, csv_path: Path | None = None):
    """Open (or create) the seeded Chroma collection.

    Returns (client, collection, embedding_name). Seeding from the
    labeled CSV happens here on first use.
    """
    from .classifier import load_labeled_rows

    rows = load_labeled_rows(csv_path)
    fit_texts = [failure_text(r["test_name"], r["message"], r["log"]) for r in rows]
    emb_name, emb_fn = get_embedding_function(fit_texts)
    client = chromadb.PersistentClient(path=str(chroma_dir or chroma_path()))
    collection = client.get_or_create_collection(
        name=collection_name(emb_name),
        embedding_function=emb_fn,  # type: ignore[arg-type]
        metadata={"hnsw:space": "cosine"},
    )
    if collection.count() == 0:
        collection.add(
            documents=fit_texts,
            metadatas=[{"test_name": r["test_name"], "category": r["category"],
                        "message": r["message"],
                        "template_family": r["template_family"]} for r in rows],
            ids=[f"row-{i}" for i in range(len(rows))],
        )
    return client, collection, emb_name


def search_similar(collection, query_text: str, k: int = 3) -> list[dict]:
    """Top-k similar past failures: dicts with test_name, category, distance."""
    if collection.count() == 0:
        return []
    result = collection.query(query_texts=[query_text], n_results=min(k, collection.count()))
    out: list[dict] = []
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    for meta, dist in zip(metadatas, distances):
        out.append({
            "test_name": (meta or {}).get("test_name", ""),
            "category": (meta or {}).get("category", ""),
            "message": (meta or {}).get("message", ""),
            "distance": float(dist) if dist is not None else 0.0,
        })
    return out


def labeled_csv_path() -> Path:
    return labeled_csv()
