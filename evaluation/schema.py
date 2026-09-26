from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class QueryCategory(str, Enum):
    """Categorization tags for evaluation queries."""
    KEYWORD = "keyword"
    NATURAL_LANGUAGE = "natural_language"
    ABBREVIATION = "abbreviation"
    TYPO = "typo"
    COURSE_CODE = "course_code"
    NUMERIC = "numeric"
    POLICY = "policy"
    ELIGIBILITY = "eligibility"
    PROCEDURE = "procedure"
    LEXICAL_DIVERSITY = "lexical_diversity"


@dataclasses.dataclass
class GoldenQuery:
    """Represents a single ground-truth evaluation item.
    
    Attributes:
        query_id: Unique identifier (e.g. 'Q001').
        query: Raw query string submitted to the search pipeline.
        category: Primary category classification.
        relevant_chunks: List of ground-truth relevant chunk IDs.
        graded_relevance: Optional mapping of chunk_id -> relevance score (e.g. 1, 2, 3).
        tags: List of descriptive tags for filtering/ablation.
        target_doc_id: Primary source document ID in the corpus.
        description: Rationale / expected answer explanation.
    """
    query_id: str
    query: str
    category: str
    relevant_chunks: List[str]
    graded_relevance: Optional[Dict[str, int]] = None
    tags: List[str] = dataclasses.field(default_factory=list)
    target_doc_id: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.query_id or not isinstance(self.query_id, str):
            raise ValueError("query_id must be a non-empty string.")
        if not self.query or not isinstance(self.query, str):
            raise ValueError("query must be a non-empty string.")
        if not self.category or not isinstance(self.category, str):
            raise ValueError("category must be a non-empty string.")
        if not isinstance(self.relevant_chunks, list):
            raise ValueError("relevant_chunks must be a list of chunk IDs.")

        # Default binary graded relevance if not provided
        if self.graded_relevance is None:
            self.graded_relevance = {cid: 1 for cid in self.relevant_chunks}

    @property
    def relevant_chunk_set(self) -> Set[str]:
        return set(self.relevant_chunks)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "category": self.category,
            "relevant_chunks": self.relevant_chunks,
            "graded_relevance": self.graded_relevance,
            "tags": self.tags,
            "target_doc_id": self.target_doc_id,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GoldenQuery:
        return cls(
            query_id=data["query_id"],
            query=data["query"],
            category=data["category"],
            relevant_chunks=data.get("relevant_chunks", []),
            graded_relevance=data.get("graded_relevance"),
            tags=data.get("tags", []),
            target_doc_id=data.get("target_doc_id"),
            description=data.get("description"),
        )


@dataclasses.dataclass
class EvaluationConfig:
    """Configuration options for evaluation execution."""
    top_k_values: List[int] = dataclasses.field(default_factory=lambda: [5, 10, 20])
    rrf_k: int = 60
    vector_match_count: int = 20
    keyword_match_count: int = 20
    max_reformulation_variants: int = 5
    primary_query_weight: float = 1.0
    use_cache: bool = True
    cache_path: Optional[str] = None


@dataclasses.dataclass
class QueryResult:
    """Per-query evaluation output."""
    query_id: str
    query: str
    category: str
    variant: str
    retrieved_chunks: List[str]
    relevant_chunks: List[str]
    metrics: Dict[str, float]
    reformulations: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "category": self.category,
            "variant": self.variant,
            "retrieved_chunks": self.retrieved_chunks,
            "relevant_chunks": self.relevant_chunks,
            "metrics": self.metrics,
            "reformulations": self.reformulations,
        }


@dataclasses.dataclass
class EvaluationSummary:
    """Summary of evaluation across a dataset."""
    variant: str
    total_queries: int
    mean_metrics: Dict[str, float]
    category_metrics: Dict[str, Dict[str, float]]
    per_query_results: List[QueryResult]
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variant": self.variant,
            "total_queries": self.total_queries,
            "mean_metrics": self.mean_metrics,
            "category_metrics": self.category_metrics,
            "per_query_results": [r.to_dict() for r in self.per_query_results],
            "metadata": self.metadata,
        }
