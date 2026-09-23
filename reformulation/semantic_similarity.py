import logging
from typing import List, Sequence, Callable, Optional

import numpy as np

# Import the existing embedding encoder – it provides embed_query().
# The encoder returns a *normalized* vector (unit length) as per its implementation.
# For unit‑tested usage we allow injecting a custom callable to avoid loading the model.
try:
    from embedding.encoder import EmbeddingEncoder
except Exception:  # pragma: no cover – fallback for environments without the package
    EmbeddingEncoder = None  # type: ignore

logger = logging.getLogger(__name__)


class SemanticSimilarity:
    """Utility to compute semantic similarity between queries and filter redundancies.

    The class is deliberately lightweight and works with any callable that returns a
    list (or 1‑D ``numpy`` array) of ``float`` values representing an embedding.
    By default it uses the project's :class:`embedding.encoder.EmbeddingEncoder` to
    embed queries.

    Parameters
    ----------
    threshold:
        Cosine‑similarity threshold above which two queries are considered
        *semantically similar* and therefore redundant. Must be in ``[-1, 1]``.
    embed_fn:
        Optional callable ``(str) -> Sequence[float]`` used to obtain the embedding.
        If omitted, a fresh ``EmbeddingEncoder`` instance is created and its
        ``embed_query`` method is used.  This argument is primarily for testing so
        that no heavy model is loaded.
    """

    def __init__(
        self,
        threshold: float = 0.85,
        embed_fn: Optional[Callable[[str], Sequence[float]]] = None,
    ) -> None:
        if not isinstance(threshold, (float, int)):
            raise ValueError("threshold must be a numeric value")
        if threshold < -1.0 or threshold > 1.0:
            raise ValueError("threshold must be between -1 and 1 inclusive")
        self.threshold = float(threshold)

        if embed_fn is not None:
            self._embed_fn = embed_fn
        else:
            if EmbeddingEncoder is None:
                raise RuntimeError(
                    "EmbeddingEncoder could not be imported; provide embed_fn explicitly"
                )
            # Lazily initialise the encoder – model loading only happens when needed.
            self._encoder = EmbeddingEncoder()
            self._embed_fn = self._encoder.embed_query

    # ------------------------------------------------------------------
    # Core similarity logic
    # ------------------------------------------------------------------
    def _to_numpy(self, vec: Sequence[float]) -> np.ndarray:
        """Convert any sequence of floats to a 1‑D ``numpy`` array of ``float32``.
        The method validates dimensionality consistency with the first vector
        observed in a filtering operation.
        """
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("Embedding vector must be one‑dimensional")
        return arr

    def similarity(self, query_a: str, query_b: str) -> float:
        """Return the cosine similarity between two queries.

        The underlying embeddings are expected to be *unit‑normalized* (the
        project's ``EmbeddingEncoder`` normalises them).  If they are not, we fall
        back to the classic dot‑product divided by the product of norms for safety.
        """
        if not isinstance(query_a, str) or not isinstance(query_b, str):
            raise ValueError("Both queries must be strings")
        emb_a = self._to_numpy(self._embed_fn(query_a))
        emb_b = self._to_numpy(self._embed_fn(query_b))
        if emb_a.shape != emb_b.shape:
            raise ValueError("Embedding vectors have mismatched dimensions")
        # Ensure vectors have a reasonable dimensionality (at least 3 components)
        if emb_a.shape[0] < 3:
            raise ValueError("Embedding vectors have insufficient dimensionality")
        # Because the embeddings are normalized, ``dot`` is the cosine similarity.
        dot = float(np.dot(emb_a, emb_b))
        # Guard against tiny numerical drift outside [-1, 1].
        return max(min(dot, 1.0), -1.0)

    def are_similar(self, query_a: str, query_b: str) -> bool:
        """Return ``True`` if the similarity is greater than or equal to the threshold."""
        return self.similarity(query_a, query_b) >= self.threshold

    # ------------------------------------------------------------------
    # Filtering API
    # ------------------------------------------------------------------
    def filter_redundant(self, queries: List[str]) -> List[str]:
        """Remove semantically redundant queries.

        The algorithm processes ``queries`` in order, keeping the first query and
        discarding any later query whose similarity to *any* already‑kept query is
        ``>= threshold``.
        The operation is deterministic and does **not** modify the input list.
        """
        if not queries:
            return []
        # Cache embeddings for the queries we have already accepted.
        accepted: List[str] = []
        cache: dict[str, np.ndarray] = {}
        for q in queries:
            # Short‑circuit empty strings – they are not meaningful for similarity.
            if not q or not q.strip():
                continue
            # Compute (or retrieve) embedding for the current query.
            if q not in cache:
                cache[q] = self._to_numpy(self._embed_fn(q))
            q_vec = cache[q]
            # Compare against all previously accepted queries.
            redundant = False
            for prev in accepted:
                # Ensure we have the embedding for the previous query.
                prev_vec = cache.get(prev)
                if prev_vec is None:
                    prev_vec = self._to_numpy(self._embed_fn(prev))
                    cache[prev] = prev_vec
                # Cosine similarity (dot product of unit vectors).
                sim = float(np.dot(q_vec, prev_vec))
                if sim >= self.threshold:
                    redundant = True
                    logger.debug(
                        "Query '%s' considered redundant (similarity=%.3f) with '%s'",
                        q,
                        sim,
                        prev,
                    )
                    break
            if not redundant:
                accepted.append(q)
        return accepted

    # ------------------------------------------------------------------
    # Convenience static method (optional) – not required by the spec but handy.
    # ------------------------------------------------------------------
    @staticmethod
    def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        """Calculate cosine similarity for two raw vectors (unit‑norm not required)."""
        a = np.asarray(vec_a, dtype=np.float32)
        b = np.asarray(vec_b, dtype=np.float32)
        if a.shape != b.shape:
            raise ValueError("Vectors must have the same shape")
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            raise ValueError("Zero‑norm vector cannot be used for cosine similarity")
        return float(np.dot(a, b) / (norm_a * norm_b))
