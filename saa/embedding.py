"""Semantic engine for equivalence chains.

Primary backend: sentence-transformers (local, offline after first download).
Fallback: TF-IDF over a fitted corpus - so the system always runs, even with
no model and no network. cosine() is the single similarity primitive used by
the whole assurance pipeline.
"""
from __future__ import annotations
import numpy as np

from . import config


class SemanticEngine:
    def __init__(self, backend: str | None = None, corpus: list[str] | None = None):
        self.backend = backend or config.EMBED_BACKEND
        self._st = None
        self._tfidf = None
        self._resolved = None
        if self.backend in ("auto", "st"):
            self._try_sentence_transformers()
        if self._st is None:
            self._init_tfidf(corpus or [])

    def _try_sentence_transformers(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._st = SentenceTransformer(config.ST_MODEL)
            self._resolved = f"sentence-transformers:{config.ST_MODEL}"
        except Exception:
            self._st = None
            if self.backend == "st":
                raise

    def _init_tfidf(self, corpus: list[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        seed = corpus or ["intent", "network", "cell", "power", "billing"]
        self._tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self._tfidf.fit(seed)
        self._resolved = "tfidf"

    @property
    def name(self) -> str:
        return self._resolved

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._st is not None:
            return np.asarray(self._st.encode(texts, normalize_embeddings=True))
        m = self._tfidf.transform(texts).toarray().astype(float)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    def cosine(self, a: str, b: str) -> float:
        v = self.encode([a, b])
        return float(np.clip(v[0] @ v[1], -1.0, 1.0))
