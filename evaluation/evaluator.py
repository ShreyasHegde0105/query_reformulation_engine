from __future__ import annotations

import dataclasses
import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from evaluation.metrics import calculate_all_metrics
from evaluation.schema import (
    EvaluationConfig,
    EvaluationSummary,
    GoldenQuery,
    QueryResult,
)
from reformulation.reformulator import QueryReformulator
from reformulation.search_pipeline import SearchPipeline
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.keyword_retriever import KeywordRetriever
from retrieval.models import RetrievedChunk
from retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger("evaluation.evaluator")


class CachedQueryReformulator:
    """Wrapper that returns pre-generated query reformulations to ensure reproducible and offline evaluation."""

    def __init__(
        self,
        cache_by_query_id: Dict[str, List[str]],
        cache_by_query_text: Optional[Dict[str, List[str]]] = None,
        fallback: Optional[QueryReformulator] = None,
    ) -> None:
        self.cache_by_query_id = cache_by_query_id
        self.cache_by_query_text = cache_by_query_text or {}
        self.fallback = fallback
        self._current_query_id: Optional[str] = None

    def set_current_query_id(self, query_id: Optional[str]) -> None:
        self._current_query_id = query_id

    def generate_variants(self, query: str, max_variants: int = 5) -> List[str]:
        # 1. Lookup by query_id if set
        if self._current_query_id and self._current_query_id in self.cache_by_query_id:
            return self.cache_by_query_id[self._current_query_id][:max_variants]

        # 2. Lookup by query string
        q_clean = query.strip().lower()
        if q_clean in self.cache_by_query_text:
            return self.cache_by_query_text[q_clean][:max_variants]

        # 3. Fallback to live reformulator if available
        if self.fallback is not None:
            try:
                return self.fallback.generate_variants(query, max_variants=max_variants)
            except Exception as exc:
                logger.warning("Fallback reformulator failed for '%s': %s", query, exc)

        return []


class RetrievalEvaluator:
    """Orchestrates retrieval quality evaluation across baseline and advanced pipeline variants.
    
    Supports:
        - Baseline A: Original Query + Vector
        - Baseline B: Original Query + Keyword/BM25
        - Baseline C: Original Query + Hybrid (RRF)
        - Variant D: Reformulation + Hybrid (No normalization/filtering)
        - Full Pipeline (Variant E): Validation -> Normalization -> Reformulation -> Semantic Filter -> Hybrid -> RRF
        - Custom user-registered ablation variants
    """

    def __init__(
        self,
        vector_retriever: Optional[VectorRetriever] = None,
        keyword_retriever: Optional[KeywordRetriever] = None,
        hybrid_retriever: Optional[HybridRetriever] = None,
        search_pipeline: Optional[SearchPipeline] = None,
        config: Optional[EvaluationConfig] = None,
        reformulation_cache: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.config = config or EvaluationConfig()
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.hybrid_retriever = hybrid_retriever
        self.search_pipeline = search_pipeline
        self.custom_variants: Dict[str, Callable[[GoldenQuery, EvaluationConfig], List[str]]] = {}

        # Load reformulation cache if provided or look for default file
        self.reformulation_cache = reformulation_cache or {}
        if not self.reformulation_cache and self.config.use_cache:
            cache_file = self.config.cache_path or os.path.join(
                os.path.dirname(__file__), "cache", "reformulations.json"
            )
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        self.reformulation_cache = json.load(f)
                    logger.info("Loaded %d cached query reformulations from %s", len(self.reformulation_cache), cache_file)
                except Exception as exc:
                    logger.warning("Failed to load reformulation cache from %s: %s", cache_file, exc)

        # Wire cached reformulator to SearchPipeline if caching enabled
        if self.config.use_cache and self.reformulation_cache and self.search_pipeline:
            cached_ref = CachedQueryReformulator(
                cache_by_query_id=self.reformulation_cache,
                fallback=getattr(self.search_pipeline, "reformulator", None),
            )
            self.cached_reformulator = cached_ref
            self.search_pipeline.reformulator = cached_ref  # type: ignore[assignment]
        else:
            self.cached_reformulator = None

    def register_variant(
        self,
        name: str,
        runner_fn: Callable[[GoldenQuery, EvaluationConfig], List[str]],
    ) -> None:
        """Register a custom retrieval variant function returning retrieved chunk IDs."""
        self.custom_variants[name] = runner_fn

    def _get_reformulations_for_query(self, query_item: GoldenQuery) -> List[str]:
        """Fetch cached or generated reformulations for a query."""
        if query_item.query_id in self.reformulation_cache:
            return self.reformulation_cache[query_item.query_id][:self.config.max_reformulation_variants]
        
        if self.search_pipeline and self.search_pipeline.reformulator:
            try:
                return self.search_pipeline.reformulator.generate_variants(
                    query_item.query,
                    max_variants=self.config.max_reformulation_variants,
                )
            except Exception as exc:
                logger.warning("LLM reformulation failed for %s: %s", query_item.query_id, exc)
        return []

    def run_query_variant(self, query_item: GoldenQuery, variant_name: str) -> List[str]:
        """Execute a specific retrieval variant for a single query.
        
        Returns:
            List of retrieved chunk IDs ordered by rank.
        """
        top_k = max(self.config.top_k_values) if self.config.top_k_values else 20

        # Set active query ID context for cached reformulator
        if self.cached_reformulator:
            self.cached_reformulator.set_current_query_id(query_item.query_id)

        # Check custom registered variants first
        if variant_name in self.custom_variants:
            return self.custom_variants[variant_name](query_item, self.config)

        if variant_name in ("baseline_vector", "original_vector"):
            if not self.vector_retriever:
                raise RuntimeError("VectorRetriever not configured for vector baseline.")
            chunks = self.vector_retriever.retrieve(
                query=query_item.query,
                match_count=top_k,
            )
            return [c.chunk_id for c in chunks]

        elif variant_name in ("baseline_keyword", "original_bm25", "original_keyword"):
            if not self.keyword_retriever:
                raise RuntimeError("KeywordRetriever not configured for keyword baseline.")
            chunks = self.keyword_retriever.retrieve(
                query=query_item.query,
                match_count=top_k,
            )
            return [c.chunk_id for c in chunks]

        elif variant_name in ("baseline_hybrid", "original_hybrid"):
            if not self.hybrid_retriever:
                raise RuntimeError("HybridRetriever not configured for hybrid baseline.")
            chunks = self.hybrid_retriever.retrieve(
                query=query_item.query,
                vector_match_count=self.config.vector_match_count,
                keyword_match_count=self.config.keyword_match_count,
                k=self.config.rrf_k,
                final_k=top_k,
            )
            return [c.chunk_id for c in chunks]

        elif variant_name in ("reformulation_hybrid", "variant_d"):
            if not self.hybrid_retriever:
                raise RuntimeError("HybridRetriever not configured for reformulation variant.")
            
            # Use raw unnormalized query + reformulations without semantic filtering
            reformulations = self._get_reformulations_for_query(query_item)
            queries_to_run = [query_item.query] + reformulations

            fused_scores: Dict[str, float] = {}
            for q in queries_to_run:
                try:
                    res = self.hybrid_retriever.retrieve(
                        query=q,
                        vector_match_count=self.config.vector_match_count,
                        keyword_match_count=self.config.keyword_match_count,
                        k=self.config.rrf_k,
                        final_k=top_k,
                    )
                    for rank, ch in enumerate(res, start=1):
                        fused_scores[ch.chunk_id] = fused_scores.get(ch.chunk_id, 0.0) + (1.0 / (self.config.rrf_k + rank))
                except Exception as exc:
                    logger.warning("Hybrid branch retrieval failed for query '%s': %s", q, exc)

            ranked_chunk_ids = sorted(fused_scores.keys(), key=lambda cid: (-fused_scores[cid], cid))
            return ranked_chunk_ids[:top_k]

        elif variant_name in ("full_pipeline", "variant_e"):
            if not self.search_pipeline:
                raise RuntimeError("SearchPipeline not configured for full pipeline evaluation.")
            
            chunks = self.search_pipeline.search(
                query=query_item.query,
                max_variants=self.config.max_reformulation_variants,
                vector_match_count=self.config.vector_match_count,
                keyword_match_count=self.config.keyword_match_count,
                k=self.config.rrf_k,
                final_k=top_k,
                primary_query_weight=self.config.primary_query_weight,
            )
            return [c.chunk_id for c in chunks]

        else:
            raise ValueError(f"Unknown retrieval variant: '{variant_name}'")

    def evaluate_query(self, query_item: GoldenQuery, variant_name: str) -> QueryResult:
        """Evaluate a single query for a specific variant."""
        retrieved_chunk_ids = self.run_query_variant(query_item, variant_name)
        reformulations = (
            self._get_reformulations_for_query(query_item)
            if variant_name in ("reformulation_hybrid", "full_pipeline", "variant_d", "variant_e")
            else None
        )

        metrics = calculate_all_metrics(
            retrieved_chunk_ids=retrieved_chunk_ids,
            relevant_chunk_ids=query_item.relevant_chunks,
            graded_relevance=query_item.graded_relevance,
            k_list=self.config.top_k_values[:2],  # [5, 10]
        )

        return QueryResult(
            query_id=query_item.query_id,
            query=query_item.query,
            category=query_item.category,
            variant=variant_name,
            retrieved_chunks=retrieved_chunk_ids,
            relevant_chunks=query_item.relevant_chunks,
            metrics=metrics,
            reformulations=reformulations,
        )

    def evaluate_variant(
        self,
        dataset: Sequence[GoldenQuery],
        variant_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationSummary:
        """Evaluate a dataset across all queries for a single variant."""
        results: List[QueryResult] = []
        for q in dataset:
            res = self.evaluate_query(q, variant_name)
            results.append(res)

        # Calculate mean metrics overall
        metric_sums: Dict[str, float] = {}
        for r in results:
            for k, v in r.metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + v

        total = len(results)
        mean_metrics = {k: (v / total if total > 0 else 0.0) for k, v in metric_sums.items()}

        # Calculate per-category breakdown
        cat_groups: Dict[str, List[QueryResult]] = {}
        for r in results:
            cat_groups.setdefault(r.category, []).append(r)

        category_metrics: Dict[str, Dict[str, float]] = {}
        for cat, cat_res in cat_groups.items():
            c_sums: Dict[str, float] = {}
            for r in cat_res:
                for k, v in r.metrics.items():
                    c_sums[k] = c_sums.get(k, 0.0) + v
            c_tot = len(cat_res)
            category_metrics[cat] = {k: (v / c_tot if c_tot > 0 else 0.0) for k, v in c_sums.items()}

        return EvaluationSummary(
            variant=variant_name,
            total_queries=total,
            mean_metrics=mean_metrics,
            category_metrics=category_metrics,
            per_query_results=results,
            metadata=metadata or {},
        )

    def evaluate_all_variants(
        self,
        dataset: Sequence[GoldenQuery],
        variant_names: Sequence[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, EvaluationSummary]:
        """Evaluate multiple variants across a dataset."""
        summaries: Dict[str, EvaluationSummary] = {}
        for name in variant_names:
            logger.info("Evaluating variant: %s on %d queries...", name, len(dataset))
            summaries[name] = self.evaluate_variant(dataset, name, metadata=metadata)
        return summaries
