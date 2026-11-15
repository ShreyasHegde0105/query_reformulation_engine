"""Production cross-encoder reranker (ranking only).

Reorders the candidate chunks produced by hybrid retrieval. Design rules:

* **Contextual passages only.** Each candidate is scored as ``Title / Section / Content``
  (:func:`reformulation.cross_encoder.build_contextual_passage`); a text-only mode is
  deliberately not offered, because it measured 0.12-0.13 NDCG@10 *worse* than no
  reranking on both query sets (audit, 2026-11-13).
* **Scores are raw logits.** They are stored as ``metadata["reranker_logit"]`` and used only
  to order candidates for the same query. They are never converted to probabilities,
  never compared with a relevance threshold and never used as relevance labels.
* **Fails safe.** Empty input returns empty output; a model load or inference error
  returns the original order (cut to ``final_k``) and reports the error.
* **Independent of evaluation.** This module does not import anything from ``evaluation``
  and has no access to the LLM judge (see ``tests/test_component_separation.py``).
* **Off by default.** Enable it with ``RERANKER_ENABLED=true`` only after a benchmark shows a
  repeatable benefit.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import Any, List, Optional, Sequence

from reformulation.cross_encoder import build_contextual_passage
from retrieval.models import RetrievedChunk

logger = logging.getLogger("reformulation.production_reranker")

__all__ = ["RerankerConfig", "RerankOutcome", "ProductionReranker", "LOGIT_KEY"]

LOGIT_KEY = "reranker_logit"
MAX_CANDIDATE_LIMIT = 100


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name, "").strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    return default


@dataclasses.dataclass(frozen=True)
class RerankerConfig:
    """Production reranker settings. There is intentionally no threshold field."""

    enabled: bool = False
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_limit: int = 50            # candidates taken from hybrid retrieval and rescored (e.g. 20, 50, 100)
    batch_size: int = 32
    max_passage_chars: int = 1500
    local_files_only: bool = True
    revision: Optional[str] = None       # pin a model snapshot for reproducibility

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_limit, int) or not 1 <= self.candidate_limit <= MAX_CANDIDATE_LIMIT:
            raise ValueError(f"candidate_limit must be an integer in 1..{MAX_CANDIDATE_LIMIT}")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_passage_chars < 100:
            raise ValueError("max_passage_chars must be >= 100")

    @classmethod
    def from_env(cls) -> "RerankerConfig":
        return cls(
            enabled=_env_bool("RERANKER_ENABLED", False),
            model_name=os.getenv("RERANKER_MODEL", "").strip() or cls.model_name,
            candidate_limit=int(os.getenv("RERANKER_CANDIDATE_LIMIT", "50")),
            batch_size=int(os.getenv("RERANKER_BATCH_SIZE", "32")),
            local_files_only=_env_bool("HF_LOCAL_FILES_ONLY", True),
            revision=os.getenv("RERANKER_REVISION", "").strip() or None,
        )


@dataclasses.dataclass
class RerankOutcome:
    """Reordered candidates plus how the reranking went."""

    results: List[RetrievedChunk]
    reranked: bool                     # False when disabled, empty, or failed (original order kept)
    latency_ms: float
    candidates_scored: int
    model_name: str
    error: Optional[str] = None


class ProductionReranker:
    """Contextual cross-encoder reranking of hybrid-retrieval candidates."""

    def __init__(self, config: Optional[RerankerConfig] = None, model: Any = None) -> None:
        self.config = config or RerankerConfig()
        self._model = model          # injectable for tests; otherwise loaded lazily

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def candidate_limit(self) -> int:
        return self.config.candidate_limit

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder
            kwargs = {"local_files_only": self.config.local_files_only}
            if self.config.revision:
                kwargs["revision"] = self.config.revision
            self._model = CrossEncoder(self.config.model_name, **kwargs)
        return self._model

    def _logits(self, query: str, candidates: Sequence[RetrievedChunk]) -> List[float]:
        model = self._load()
        pairs = [(query, build_contextual_passage(c, max_characters=self.config.max_passage_chars)) for c in candidates]
        out: List[float] = []
        for i in range(0, len(pairs), self.config.batch_size):
            out.extend(float(s) for s in model.predict(pairs[i:i + self.config.batch_size]))
        if len(out) != len(candidates):
            raise RuntimeError(f"model returned {len(out)} scores for {len(candidates)} candidates")
        return out

    def rerank(self, query: str, candidates: Sequence[RetrievedChunk], final_k: Optional[int] = None) -> RerankOutcome:
        """Reorder the first ``candidate_limit`` candidates by cross-encoder logit.

        Candidates beyond ``candidate_limit`` are not rescored and are dropped. Returned chunks are
        copies; their ``score`` (the retrieval fusion score) is left unchanged and the logit is added
        as ``metadata["reranker_logit"]``.
        """
        t0 = time.perf_counter()
        pool = list(candidates)[: self.config.candidate_limit]
        k = len(pool) if final_k is None else max(0, int(final_k))

        def done(results, reranked, error=None, scored=0):
            return RerankOutcome(results=results[:k], reranked=reranked, error=error, candidates_scored=scored,
                                 latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                                 model_name=self.config.model_name)

        if not pool:
            return done([], False)
        if not isinstance(query, str) or not query.strip():
            return done(pool, False, error="empty query; original order kept")
        try:
            logits = self._logits(query, pool)
        except Exception as exc:  # model missing, OOM, tokenizer error ...
            logger.error("Reranking failed (%s); keeping retrieval order", exc)
            return done(pool, False, error=f"{type(exc).__name__}: {exc}")

        scored = []
        for chunk, logit in zip(pool, logits):
            copy = dataclasses.replace(chunk, metadata={**(chunk.metadata or {}), LOGIT_KEY: logit})
            scored.append(copy)
        # Stable on ties: retrieval order decides.
        order = sorted(range(len(scored)), key=lambda i: (-scored[i].metadata[LOGIT_KEY], i))
        return done([scored[i] for i in order], True, scored=len(scored))
