"""Cross-encoder reranker for second-stage candidate pool reranking.

This module provides a standalone CrossEncoderReranker that:
- Loads lazily on first use (avoid cost at import time)
- Scores (query, chunk_text) pairs in batches
- Returns candidates re-sorted by cross-encoder score (descending)
- Stores ce_score in chunk.metadata["ce_score"]
- Is configurable via environment variables
- Degrades gracefully when disabled or model unavailable

Environment Variables:
    RERANKER_ENABLED:       "true" / "false" (default: "false")
    RERANKER_MODEL:         HF model identifier (default: "cross-encoder/ms-marco-MiniLM-L-6-v2")
    RERANKER_CANDIDATE_K:   how many candidates to collect before reranking (default: 50)
    RERANKER_BATCH_SIZE:    inference batch size (default: 32)
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.models import RetrievedChunk

logger = logging.getLogger("reformulation.cross_encoder")

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_DEFAULT_CANDIDATE_K = 50
_DEFAULT_BATCH_SIZE = 32


def _env_bool(var: str, default: bool) -> bool:
    """Read a boolean environment variable."""
    val = os.environ.get(var, "").strip().lower()
    if val in ("1", "true", "yes"):
        return True
    if val in ("0", "false", "no"):
        return False
    return default


def _env_int(var: str, default: int) -> int:
    """Read an integer environment variable."""
    val = os.environ.get(var, "").strip()
    try:
        return int(val) if val else default
    except ValueError:
        logger.warning("Invalid integer value for %s: %r. Using default %d.", var, val, default)
        return default



def build_contextual_passage(chunk: "RetrievedChunk", max_characters: int = 1500) -> str:
    """Format chunk into a structured contextual passage for cross-encoder reranking.

    Structure:
        Title: <title>
        Section: <section>
        Content: <chunk_text>

    Handles missing titles and sections safely without emitting 'None'. Applies safe character truncation.
    """
    parts: list[str] = []
    if chunk.title and chunk.title.strip():
        parts.append(f"Title: {chunk.title.strip()}")
    if chunk.section and chunk.section.strip():
        parts.append(f"Section: {chunk.section.strip()}")

    content = (chunk.chunk_text or "").strip()
    if parts:
        parts.append(f"Content: {content}")
        passage = "\n".join(parts)
    else:
        passage = content

    if len(passage) > max_characters:
        passage = passage[:max_characters].rsplit(" ", 1)[0] + "..."
    return passage


class CrossEncoderReranker:
    """Second-stage cross-encoder reranker for candidate pool reranking.

    Usage::

        reranker = CrossEncoderReranker()          # lazy — no model loaded yet
        reranked = reranker.rerank(query, chunks)  # model loaded on first call

    The reranker is a **pure ranking stage**: it does NOT retrieve new candidates.
    It only re-orders the chunks already present in the input candidate pool.

    Args:
        model_name:  HuggingFace cross-encoder model identifier.
        batch_size:  Number of (query, chunk) pairs scored in a single forward pass.
        enabled:     If False, rerank() returns candidates unchanged (passthrough).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        enabled: bool = True,
        contextual: bool = True,
        max_passage_chars: int = 1500,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.enabled = enabled
        self.contextual = contextual
        self.max_passage_chars = max_passage_chars
        self._model: Optional[object] = None  # lazy — loaded on first rerank() call

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the cross-encoder model (called once, on first use)."""
        if self._model is not None:
            return
        logger.info("Loading cross-encoder model: %s", self.model_name)
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded successfully: %s", self.model_name)
        except ImportError as exc:
            raise ImportError(
                "sentence_transformers is required for CrossEncoderReranker. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        except Exception as exc:
            logger.error("Failed to load cross-encoder model %s: %s", self.model_name, exc)
            raise

    @property
    def is_loaded(self) -> bool:
        """True if the underlying model has been loaded into memory."""
        return self._model is not None

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: "List[RetrievedChunk]",
        top_k: Optional[int] = None,
        contextual: Optional[bool] = None,
    ) -> "List[RetrievedChunk]":
        """Rerank ``candidates`` using cross-encoder scores for ``query``.

        The method:
        1. Slices candidates to top_k pool (if given), preserving remainder unranked.
        2. Builds contextual pairs (Title + Section + Content) or text-only pairs.
        3. Runs forward-pass inference in batches of ``self.batch_size``.
        4. Attaches ``ce_score`` to each chunk's ``metadata`` dict.
        5. Returns a new list sorted by ``ce_score`` descending followed by tail.

        If ``self.enabled`` is False, returns candidates unchanged (no model load).
        If ``candidates`` is empty, returns an empty list immediately.

        Args:
            query:      The user/reformulated query string.
            candidates: Ordered list of RetrievedChunk objects from the retrieval stage.
            top_k:      Maximum candidates from the front of the list to rerank.
            contextual: Whether to format contextual passages (defaults to self.contextual).

        Returns:
            New list of RetrievedChunk objects sorted by cross-encoder score (desc).
            Original ``rrf_score`` (``chunk.score``) is preserved in all returned chunks.
        """
        if not self.enabled:
            logger.debug("CrossEncoderReranker is disabled. Returning candidates unchanged.")
            return list(candidates)

        if not candidates:
            return []

        use_contextual = self.contextual if contextual is None else contextual
        rerank_pool = candidates[:top_k] if top_k is not None else candidates
        unranked_tail = candidates[top_k:] if top_k is not None else []

        if not rerank_pool:
            return list(candidates)

        # Lazy model load
        self._load_model()

        # Build (query, passage) pairs
        if use_contextual:
            pairs = [
                (query, build_contextual_passage(chunk, max_characters=self.max_passage_chars))
                for chunk in rerank_pool
            ]
        else:
            pairs = [(query, chunk.chunk_text or "") for chunk in rerank_pool]

        # Score in batches with safe error recovery
        try:
            scores = self._score_pairs(pairs)
        except Exception as exc:
            logger.error(
                "CrossEncoder scoring failed for query %r: %s. Returning input candidate order.",
                query[:80],
                exc,
            )
            return list(candidates)

        # Attach ce_score to metadata (copy to avoid mutating shared state)
        import copy
        scored_chunks = []
        for chunk, ce_score in zip(rerank_pool, scores):
            chunk_copy = copy.copy(chunk)
            chunk_copy.metadata = dict(chunk.metadata or {})
            chunk_copy.metadata["ce_score"] = float(ce_score)
            scored_chunks.append(chunk_copy)

        # Sort by cross-encoder score descending; use chunk_id as tie-breaker for determinism
        scored_chunks.sort(key=lambda c: (-c.metadata["ce_score"], c.chunk_id))

        return scored_chunks + unranked_tail

    def score(
        self,
        query: str,
        candidates: "List[RetrievedChunk]",
        contextual: Optional[bool] = None,
    ) -> List[float]:
        """Return raw cross-encoder relevance scores for ``candidates``, in input order.

        Unlike ``rerank()``, this ignores ``self.enabled`` so the model can serve as
        a label-free relevance judge (e.g. for queries without ground truth).
        """
        if not candidates:
            return []
        self._load_model()
        use_contextual = self.contextual if contextual is None else contextual
        if use_contextual:
            pairs = [
                (query, build_contextual_passage(chunk, max_characters=self.max_passage_chars))
                for chunk in candidates
            ]
        else:
            pairs = [(query, chunk.chunk_text or "") for chunk in candidates]
        return self._score_pairs(pairs)

    def _score_pairs(self, pairs: List[tuple]) -> List[float]:
        """Score a list of (query, passage) pairs using the cross-encoder.

        Splits into batches of ``self.batch_size`` to control memory usage.

        Args:
            pairs: List of (query_str, passage_str) tuples.

        Returns:
            List of float scores, one per pair, in the same order.
        """
        all_scores: List[float] = []
        model = self._model  # type: ignore[assignment]

        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            batch_scores = model.predict(batch)  # type: ignore[union-attr]
            # predict() may return a numpy array; convert to Python floats
            all_scores.extend(float(s) for s in batch_scores)

        return all_scores

    # ------------------------------------------------------------------
    # Factory / convenience
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "CrossEncoderReranker":
        """Construct a CrossEncoderReranker from environment variables.

        Reads:
            RERANKER_ENABLED     (default: false)
            RERANKER_MODEL       (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
            RERANKER_BATCH_SIZE  (default: 32)

        Returns:
            CrossEncoderReranker configured from the environment.
        """
        enabled = _env_bool("RERANKER_ENABLED", default=False)
        model_name = os.environ.get("RERANKER_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
        batch_size = _env_int("RERANKER_BATCH_SIZE", default=_DEFAULT_BATCH_SIZE)

        logger.info(
            "CrossEncoderReranker.from_env() → enabled=%s model=%s batch_size=%d",
            enabled,
            model_name,
            batch_size,
        )
        return cls(model_name=model_name, batch_size=batch_size, enabled=enabled)

    def __repr__(self) -> str:
        return (
            f"CrossEncoderReranker("
            f"model={self.model_name!r}, "
            f"batch_size={self.batch_size}, "
            f"enabled={self.enabled}, "
            f"loaded={self.is_loaded})"
        )
