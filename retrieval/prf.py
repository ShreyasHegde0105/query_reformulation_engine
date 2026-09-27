"""Pseudo-relevance feedback (RM3) over the in-process BM25 index.

A non-LLM query-expansion baseline: retrieve with the original query, assume the
top documents are relevant, and add the terms that are most characteristic of
them to the query.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

from retrieval.bm25_retriever import BM25Retriever, tokenize
from retrieval.models import RetrievedChunk


def rm3_term_weights(
    query: str,
    bm25: BM25Retriever,
    fb_docs: int = 10,
    fb_terms: int = 10,
    original_query_weight: float = 0.5,
) -> Dict[str, float]:
    """RM3 expanded query: ``λ·P(w|Q) + (1-λ)·P(w|R)``.

    ``P(w|R)`` is the relevance model estimated from the top ``fb_docs`` BM25
    results (each document's term distribution weighted by its normalized
    score), truncated to the ``fb_terms`` heaviest terms and renormalized.
    """
    if fb_docs < 1 or fb_terms < 1:
        raise ValueError("fb_docs and fb_terms must be >= 1")
    if not 0.0 <= original_query_weight <= 1.0:
        raise ValueError("original_query_weight must be in [0, 1]")

    q_terms = tokenize(query)
    if not q_terms:
        return {}
    q_counts = Counter(q_terms)
    p_q = {t: c / len(q_terms) for t, c in q_counts.items()}

    feedback = bm25.retrieve_weighted({t: 1.0 for t in q_counts}, match_count=fb_docs)
    total_score = sum(c.score for c in feedback)
    if not feedback or total_score <= 0:
        return p_q

    relevance: Counter = Counter()
    for chunk in feedback:
        terms = bm25.doc_terms(chunk.chunk_id)
        length = sum(terms.values())
        if not length:
            continue
        doc_weight = chunk.score / total_score
        for term, tf in terms.items():
            relevance[term] += doc_weight * tf / length

    top = relevance.most_common(fb_terms)
    norm = sum(w for _, w in top)
    p_r = {t: w / norm for t, w in top} if norm > 0 else {}

    lam = original_query_weight
    return {t: lam * p_q.get(t, 0.0) + (1 - lam) * p_r.get(t, 0.0) for t in set(p_q) | set(p_r)}


class RM3Retriever:
    """Keyword retriever that expands each query with RM3 before BM25 scoring.

    Same ``retrieve`` contract as :class:`BM25Retriever`, so it can be used as the
    keyword branch of :class:`~retrieval.hybrid_retriever.HybridRetriever`.
    """

    def __init__(self, bm25: BM25Retriever, fb_docs: int = 10, fb_terms: int = 10,
                 original_query_weight: float = 0.5) -> None:
        self.bm25 = bm25
        self.fb_docs = fb_docs
        self.fb_terms = fb_terms
        self.original_query_weight = original_query_weight

    def retrieve(self, query: str, match_count: int = 20, filter_domain: Optional[str] = None) -> List[RetrievedChunk]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        weights = rm3_term_weights(query, self.bm25, self.fb_docs, self.fb_terms, self.original_query_weight)
        if not weights:
            return []
        return self.bm25.retrieve_weighted(weights, match_count=match_count, filter_domain=filter_domain)
