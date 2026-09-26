"""Retrieval Quality Evaluation Framework for Query Reformulation Engine.

Provides evaluation dataset schema, standard IR metrics (Recall@K, MRR, NDCG@K),
multi-variant retrieval evaluation orchestration, and reporting.
"""

from evaluation.schema import (
    GoldenQuery,
    EvaluationConfig,
    QueryResult,
    EvaluationSummary,
)
from evaluation.metrics import (
    calculate_recall_at_k,
    calculate_mrr,
    calculate_dcg_at_k,
    calculate_ndcg_at_k,
    calculate_all_metrics,
)
from evaluation.evaluator import RetrievalEvaluator

__all__ = [
    "GoldenQuery",
    "EvaluationConfig",
    "QueryResult",
    "EvaluationSummary",
    "calculate_recall_at_k",
    "calculate_mrr",
    "calculate_dcg_at_k",
    "calculate_ndcg_at_k",
    "calculate_all_metrics",
    "RetrievalEvaluator",
]
