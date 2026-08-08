"""
Pluggable embedding backends.

- "chroma_default": Chroma's built-in ONNX MiniLM embedding function.
  Best quality/cost tradeoff for this use case, but downloads a small
  model file on first use — requires outbound network access in your
  deployment environment.
- "tfidf": pure-local scikit-learn TF-IDF vectorizer. No downloads, so
  it's useful to smoke-test the ingestion pipeline end-to-end in
  network-restricted environments, but it's a bag-of-words model, not
  a semantic embedding — swap to chroma_default (or an API-based
  embedding model) before relying on it for real retrieval quality.
"""
from __future__ import annotations

from chromadb.utils import embedding_functions


def get_embedding_function(backend: str):
    if backend == "chroma_default":
        return embedding_functions.DefaultEmbeddingFunction()
    if backend == "tfidf":
        return _TfidfEmbeddingFunction()
    raise ValueError(f"Unknown embedding backend: {backend}")


class _TfidfEmbeddingFunction:
    """Fit-on-first-call TF-IDF embedding function compatible with
    Chroma's EmbeddingFunction interface (a callable taking a list of
    strings and returning a list of float vectors).

    Unlike a real semantic embedding model, TF-IDF has no meaning outside
    its fitted vocabulary, and Chroma may reconstruct a fresh instance for
    each process (e.g. ingest.py vs. a separate query script). So the
    fitted vectorizer is pickled to disk on first fit and reloaded on
    reconstruction, keeping ingestion-time and query-time vocab in sync.
    """

    _CACHE_PATH = "./chroma_store/_tfidf_vectorizer.joblib"

    def __init__(self, max_features: int = 2048):
        from pathlib import Path

        from sklearn.feature_extraction.text import TfidfVectorizer

        self.max_features = max_features
        cache = Path(self._CACHE_PATH)
        if cache.exists():
            import joblib

            self._vectorizer = joblib.load(cache)
            self._fitted = True
        else:
            self._vectorizer = TfidfVectorizer(max_features=max_features)
            self._fitted = False

    def __call__(self, input: list[str]) -> list[list[float]]:
        if not self._fitted:
            matrix = self._vectorizer.fit_transform(input)
            self._fitted = True
            self._save()
        else:
            matrix = self._vectorizer.transform(input)
        return matrix.toarray().tolist()

    def embed_query(self, input: list[str]) -> list[list[float]]:
        # Never re-fit on a query — always use the vocabulary learned
        # during ingestion, even if this instance hasn't seen it fit.
        if not self._fitted:
            raise RuntimeError(
                "TF-IDF vectorizer has not been fitted yet — run ingestion "
                "before querying."
            )
        return self._vectorizer.transform(input).toarray().tolist()

    def _save(self) -> None:
        import joblib
        from pathlib import Path

        Path(self._CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._vectorizer, self._CACHE_PATH)

    @staticmethod
    def name() -> str:
        return "tfidf_local"

    def get_config(self) -> dict:
        return {"max_features": self.max_features}

    @staticmethod
    def build_from_config(config: dict) -> "_TfidfEmbeddingFunction":
        return _TfidfEmbeddingFunction(max_features=config.get("max_features", 2048))
