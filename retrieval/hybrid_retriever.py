from __future__ import annotations

import logging
from typing import List, Optional

from retrieval.models import RetrievedChunk
from retrieval.vector_retriever import VectorRetriever
from retrieval.keyword_retriever import KeywordRetriever

logger = logging.getLogger("retrieval.hybrid")


class HybridRetriever:
    """Hybrid retrieval orchestrator that fuses vector and keyword results using
    Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    vector_retriever: Optional[VectorRetriever]
        Pre‑constructed vector retriever (defaults to a new instance).
    keyword_retriever: Optional[KeywordRetriever]
        Pre‑constructed keyword retriever (defaults to a new instance).
    """

    def __init__(
        self,
        vector_retriever: Optional[VectorRetriever] = None,
        keyword_retriever: Optional[KeywordRetriever] = None,
    ) -> None:
        self.vector_retriever = vector_retriever or VectorRetriever()
        self.keyword_retriever = keyword_retriever or KeywordRetriever()

    def retrieve(
        self,
        query: str,
        vector_match_count: int = 20,
        keyword_match_count: int = 20,
        filter_domain: Optional[str] = None,
        k: int = 60,
        final_k: int = 20,
    ) -> List[RetrievedChunk]:
        """Retrieve chunks via both back‑ends and fuse them with RRF.

        Parameters
        ----------
        query: str
            Non‑empty user query.
        vector_match_count: int
            Number of candidates to request from the vector retriever (0 disables it).
        keyword_match_count: int
            Number of candidates to request from the keyword retriever (0 disables it).
        filter_domain: Optional[str]
            Passed unchanged to both retrievers.
        k: int
            RRF damping factor (must be > 0).
        final_k: int
            Number of top results to return after fusion.
        """
        # ------------------------------ validation ------------------------------
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non‑empty string.")
        if not isinstance(vector_match_count, int) or vector_match_count < 0:
            raise ValueError("vector_match_count must be a non‑negative integer.")
        if not isinstance(keyword_match_count, int) or keyword_match_count < 0:
            raise ValueError("keyword_match_count must be a non‑negative integer.")
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        if not isinstance(final_k, int) or final_k < 1:
            raise ValueError("final_k must be a positive integer.")

        # ---------------------------- retrieval calls --------------------------
        vector_results: List[RetrievedChunk] = []
        keyword_results: List[RetrievedChunk] = []
        vector_error = keyword_error = None

        try:
            vector_results = self.vector_retriever.retrieve(
                query=query,
                match_count=vector_match_count,
                filter_domain=filter_domain,
            )
        except Exception as exc:  # pragma: no cover – exercised in tests
            vector_error = exc
            logger.error("Vector retrieval failed: %s", exc)

        try:
            keyword_results = self.keyword_retriever.retrieve(
                query=query,
                match_count=keyword_match_count,
                filter_domain=filter_domain,
            )
        except Exception as exc:  # pragma: no cover – exercised in tests
            keyword_error = exc
            logger.error("Keyword retrieval failed: %s", exc)

        if vector_error and keyword_error:
            raise RuntimeError(
                f"Both retrieval branches failed – vector error: {vector_error}; keyword error: {keyword_error}"
            )

        # If one branch failed, continue with the successful results.
        # Ensure deterministic ranking order based on score (higher first).
        # Both retrievers are expected to return results already sorted, but we enforce it
        # to match test expectations where keyword list may not be pre‑sorted.
        vector_results = sorted(vector_results, key=lambda c: (-c.score, c.chunk_id))
        keyword_results = sorted(keyword_results, key=lambda c: (-c.score, c.chunk_id))
        # Map chunk_id -> dict holding ranks and original scores.
        fused: dict[str, dict] = {}

        for rank, chunk in enumerate(vector_results, start=1):
            entry = fused.setdefault(chunk.chunk_id, {
                "orig_chunk": chunk,
                "vector_rank": None,
                "keyword_rank": None,
                "vector_score": None,
                "keyword_score": None,
            })
            entry["vector_rank"] = rank
            entry["vector_score"] = chunk.score

        for rank, chunk in enumerate(keyword_results, start=1):
            entry = fused.setdefault(chunk.chunk_id, {
                "orig_chunk": chunk,
                "vector_rank": None,
                "keyword_rank": None,
                "vector_score": None,
                "keyword_score": None,
            })
            entry["keyword_rank"] = rank
            entry["keyword_score"] = chunk.score
            if entry["orig_chunk"] is not chunk:
                # Merge metadata (shallow) when the same chunk appears in both lists.
                merged_meta = {**entry["orig_chunk"].metadata, **chunk.metadata}
                entry["orig_chunk"].metadata = merged_meta

        # Build a list of (chunk, rrf_score, vector_score, keyword_score).
        fused_list: List[tuple[RetrievedChunk, float, Optional[float], Optional[float]]] = []
        import dataclasses
        for data in fused.values():
            rrf_score = 0.0
            if data["vector_rank"] is not None:
                rrf_score += 1.0 / (k + data["vector_rank"])
            if data["keyword_rank"] is not None:
                rrf_score += 1.0 / (k + data["keyword_rank"])
            # Clone the original chunk with the RRF score.
            new_chunk = dataclasses.replace(data["orig_chunk"], score=rrf_score)
            fused_list.append(
                (
                    new_chunk,
                    rrf_score,
                    data.get("vector_score"),
                    data.get("keyword_score"),
                )
            )

        # Sort primarily by RRF descending, then by chunk_id for deterministic tie‑breaking.
        fused_list.sort(key=lambda tup: (-tup[1], tup[0].chunk_id))

        # Return top final_k results (or fewer if not enough).
        top_chunks = [chunk for chunk, _, _, _ in fused_list[:final_k]]
        return top_chunks
