"""MiniLM / TF-IDF embedding switch tests.

- TF-IDF vectors are consistent in dimension and L2-normalised.
- If MiniLM cannot be built, get_embedding_function falls back to
  TF-IDF (tested by forcing the MiniLM constructor to raise).
- The real MiniLM path is tested only when sentence-transformers is
  actually installed (otherwise skipped, not failed).
"""
import numpy as np
import pytest

from app.embeddings import TfidfEmbeddingFunction, get_embedding_function

TEXTS = [
    "Connection refused to database at db.internal:5432",
    "NullPointerException at Cart.total",
    "StaleElementReferenceException: element is not attached",
]


def test_tfidf_vectors_consistent_and_normalised():
    fn = TfidfEmbeddingFunction(TEXTS)
    vectors = fn.embed_documents(TEXTS)
    assert len(vectors) == 3
    dims = {len(v) for v in vectors}
    assert len(dims) == 1
    for v in vectors:
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-6
    assert fn.embed_query(TEXTS[:1])[0] == vectors[0]
    assert fn.name() == "tfidf"
    assert fn.is_legacy() is False


def test_fallback_when_minilm_fails(monkeypatch):
    import app.embeddings as emb_mod

    def boom():
        raise RuntimeError("no model here")

    monkeypatch.setattr(emb_mod, "MiniLMEmbeddingFunction", boom)
    name, fn = get_embedding_function(TEXTS)
    assert name == "tfidf"
    assert fn.embed_documents(TEXTS[:1])


try:
    import sentence_transformers  # noqa: F401
    HAS_MINILM = True
except ImportError:
    HAS_MINILM = False


@pytest.mark.skipif(not HAS_MINILM, reason="sentence-transformers not installed")
def test_minilm_real_path_when_available():
    name, fn = get_embedding_function(TEXTS)
    assert name == "minilm"
    vectors = fn.embed_documents(TEXTS)
    assert len(vectors) == 3
    assert all(len(v) == 384 for v in vectors)
    assert fn.name() == "minilm"
