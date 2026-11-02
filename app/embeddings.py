"""Embedding functions for ChromaDB.

Preferred: sentence-transformers all-MiniLM-L6-v2. If that package or
model cannot be loaded in this environment (not installed, download
fails, out of memory), fall back to a scikit-learn TF-IDF embedding
function fitted on the labeled data, and say so clearly. The caller gets
(name, function) so RESULTS.md can report which one was actually used.

Both functions satisfy ChromaDB's current custom embedding-function
interface (embed_documents / embed_query, plus __call__ and a stable
name()). The collection name includes the embedding name, so a TF-IDF
collection and a MiniLM collection never collide.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfEmbeddingFunction:
    """TF-IDF vectors (L2-normalised, so cosine works in Chroma).

    Implements ChromaDB's current embedding-function interface
    (embed_documents / embed_query) plus __call__ for older callers.
    """

    def __init__(self, texts: list[str]) -> None:
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._vectorizer.fit(texts)

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        matrix = self._vectorizer.transform(input)
        return matrix.toarray().astype(float).tolist()

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)

    def name(self) -> str:
        return "tfidf"

    def is_legacy(self) -> bool:
        return False


class MiniLMEmbeddingFunction:
    """sentence-transformers all-MiniLM-L6-v2 embeddings."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_documents(self, input: list[str]) -> list[list[float]]:
        vectors = self._model.encode(input, normalize_embeddings=True)
        return np.asarray(vectors, dtype=float).tolist()

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)

    def __call__(self, input: list[str]) -> list[list[float]]:
        return self.embed_documents(input)

    def name(self) -> str:
        return "minilm"

    def is_legacy(self) -> bool:
        return False


def get_embedding_function(fit_texts: list[str]) -> tuple[str, object]:
    """Try MiniLM first; fall back to TF-IDF on any failure.

    Returns (name, function). Never raises for a missing model: the
    fallback is the documented behaviour, not an error.
    """
    try:
        fn = MiniLMEmbeddingFunction()
        # Prove the model actually loads by embedding one short text.
        fn(["connectivity check"])
        return "minilm", fn
    except Exception:
        return "tfidf", TfidfEmbeddingFunction(fit_texts)
