from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Set


def _deduplicate_preserve_order(items: Sequence[str]) -> List[str]:
    """Deduplicate a sequence of items while preserving the earliest appearance order."""
    seen: Set[str] = set()
    result: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def calculate_recall_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Set[str] | Sequence[str],
    k: int,
) -> float:
    """Calculate Recall@K for a retrieved result list against ground-truth relevant chunks.
    
    Formula:
        Recall@K = |top_k(retrieved) ∩ relevant| / |relevant|
        
    Edge Cases:
        - If relevant set is empty: returns 0.0 (avoids division by zero).
        - If retrieved list is empty: returns 0.0.
        - If k <= 0: returns 0.0.
        - Duplicates in retrieved list are deduplicated preserving first-seen order before top-k slicing.
        
    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        relevant_chunk_ids: Set or list of ground-truth relevant chunk IDs.
        k: Cutoff rank for evaluation (e.g. 5, 10, 20).
        
    Returns:
        Recall score between 0.0 and 1.0.
    """
    if k <= 0:
        return 0.0

    relevant_set = set(relevant_chunk_ids)
    if not relevant_set:
        return 0.0

    if not retrieved_chunk_ids:
        return 0.0

    deduped_retrieved = _deduplicate_preserve_order(retrieved_chunk_ids)
    top_k_retrieved = set(deduped_retrieved[:k])
    
    hits = len(top_k_retrieved.intersection(relevant_set))
    return float(hits) / float(len(relevant_set))


def calculate_mrr(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Set[str] | Sequence[str],
) -> float:
    """Calculate Reciprocal Rank (RR) of the first relevant chunk in the retrieved list.
    
    Formula:
        RR = 1 / rank_of_first_relevant_chunk (1-indexed)
        RR = 0.0 if no relevant chunk is retrieved.
        
    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        relevant_chunk_ids: Set or list of ground-truth relevant chunk IDs.
        
    Returns:
        Reciprocal rank between 0.0 and 1.0.
    """
    relevant_set = set(relevant_chunk_ids)
    if not relevant_set or not retrieved_chunk_ids:
        return 0.0

    deduped_retrieved = _deduplicate_preserve_order(retrieved_chunk_ids)
    for rank, chunk_id in enumerate(deduped_retrieved, start=1):
        if chunk_id in relevant_set:
            return 1.0 / float(rank)

    return 0.0


def calculate_dcg_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevance_map: Dict[str, float | int],
    k: int,
) -> float:
    """Calculate Discounted Cumulative Gain at rank K (DCG@K).
    
    Formula:
        DCG@K = sum_{i=1}^{min(k, |retrieved|)} (2^{rel_i} - 1) / log2(i + 1)
        
    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        relevance_map: Mapping of chunk_id -> numeric relevance score (e.g. 0, 1, 2, 3).
        k: Cutoff rank.
        
    Returns:
        DCG score >= 0.0.
    """
    if k <= 0 or not retrieved_chunk_ids or not relevance_map:
        return 0.0

    deduped_retrieved = _deduplicate_preserve_order(retrieved_chunk_ids)
    dcg = 0.0
    for rank, chunk_id in enumerate(deduped_retrieved[:k], start=1):
        rel = float(relevance_map.get(chunk_id, 0.0))
        if rel > 0.0:
            dcg += (math.pow(2.0, rel) - 1.0) / math.log2(rank + 1.0)

    return dcg


def calculate_ndcg_at_k(
    retrieved_chunk_ids: Sequence[str],
    relevance_map: Dict[str, float | int],
    k: int,
) -> float:
    """Calculate Normalized Discounted Cumulative Gain at rank K (NDCG@K).
    
    Formula:
        NDCG@K = DCG@K / IDCG@K
        where IDCG@K is the maximum possible DCG@K from the ground-truth relevance labels.
        
    Args:
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs.
        relevance_map: Mapping of chunk_id -> numeric relevance score.
        k: Cutoff rank.
        
    Returns:
        NDCG score between 0.0 and 1.0.
    """
    if k <= 0 or not relevance_map:
        return 0.0

    # Calculate Ideal DCG (IDCG) by sorting ground-truth relevance scores descending
    positive_rels = sorted(
        [float(rel) for rel in relevance_map.values() if float(rel) > 0.0],
        reverse=True,
    )
    if not positive_rels:
        return 0.0

    idcg = 0.0
    for rank, rel in enumerate(positive_rels[:k], start=1):
        idcg += (math.pow(2.0, rel) - 1.0) / math.log2(rank + 1.0)

    if idcg <= 0.0:
        return 0.0

    dcg = calculate_dcg_at_k(retrieved_chunk_ids, relevance_map, k)
    return dcg / idcg


def calculate_all_metrics(
    retrieved_chunk_ids: Sequence[str],
    relevant_chunk_ids: Sequence[str] | Set[str],
    graded_relevance: Optional[Dict[str, float | int]] = None,
    k_list: Sequence[int] = (5, 10),
) -> Dict[str, float]:
    """Calculate all standard retrieval metrics for a single query result.
    
    Computes:
        - recall@{k} for each k in k_list
        - mrr
        - ndcg@{k} for each k in k_list
        
    Args:
        retrieved_chunk_ids: List of retrieved chunk IDs in ranking order.
        relevant_chunk_ids: Ground truth relevant chunk IDs.
        graded_relevance: Optional dictionary mapping chunk_id -> score.
        k_list: List of cutoffs (default [5, 10]).
        
    Returns:
        Dictionary of metric name -> computed score.
    """
    rel_set = set(relevant_chunk_ids)
    rel_map: Dict[str, float | int]
    if graded_relevance is not None and len(graded_relevance) > 0:
        rel_map = graded_relevance
    else:
        rel_map = {cid: 1 for cid in rel_set}

    results: Dict[str, float] = {}

    for k in k_list:
        results[f"recall@{k}"] = calculate_recall_at_k(retrieved_chunk_ids, rel_set, k)

    results["mrr"] = calculate_mrr(retrieved_chunk_ids, rel_set)

    for k in k_list:
        results[f"ndcg@{k}"] = calculate_ndcg_at_k(retrieved_chunk_ids, rel_map, k)

    return results
