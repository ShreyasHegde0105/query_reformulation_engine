import logging
import dataclasses
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from reformulation.validator import validate_query
from reformulation.normalizer import normalize_query
from reformulation.reformulator import QueryReformulator
from reformulation.semantic_similarity import SemanticSimilarity
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SearchResult:
    """Ranked results plus a trace of how they were produced."""

    results: List[RetrievedChunk]
    original_query: str
    normalized_query: str
    reformulations: List[str]
    queries_used: List[str]
    timings_ms: Dict[str, float]
    reformulation_error: Optional[str] = None
    reformulation_cached: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "normalized_query": self.normalized_query,
            "reformulations": self.reformulations,
            "queries_used": self.queries_used,
            "reformulation_error": self.reformulation_error,
            "reformulation_cached": self.reformulation_cached,
            "timings_ms": self.timings_ms,
            "results": [c.to_dict() for c in self.results],
        }


class SearchPipeline:
    """End-to-end query processing and retrieval pipeline.

    Orchestrates validation, normalization, reformulation, semantic filtering,
    and multi-query hybrid retrieval aggregation.
    """

    def __init__(
        self,
        reformulator: QueryReformulator,
        semantic_similarity: SemanticSimilarity,
        hybrid_retriever: HybridRetriever,
        max_workers: int = 4,
        reformulation_cache_size: int = 256,
    ):
        """Initialize the pipeline with its required dependencies.

        Parameters
        ----------
        reformulator: QueryReformulator
            Generates semantically equivalent search alternatives.
        semantic_similarity: SemanticSimilarity
            Filters out highly redundant query reformulations.
        hybrid_retriever: HybridRetriever
            Executes retrieval against vector and keyword backends.
        max_workers: int
            Threads used to retrieve the query variants concurrently (1 = sequential).
        reformulation_cache_size: int
            Number of recent queries whose reformulations are kept in memory
            (0 disables caching). Avoids a repeat LLM call for a repeated query.
        """
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if reformulation_cache_size < 0:
            raise ValueError("reformulation_cache_size must be >= 0")
        self.reformulator = reformulator
        self.semantic_similarity = semantic_similarity
        self.hybrid_retriever = hybrid_retriever
        self.max_workers = max_workers
        self.reformulation_cache_size = reformulation_cache_size
        self._reformulation_cache: "OrderedDict[tuple[str, int], List[str]]" = OrderedDict()

    def search(
        self,
        query: str,
        max_variants: int = 5,
        vector_match_count: int = 20,
        keyword_match_count: int = 20,
        filter_domain: Optional[str] = None,
        k: int = 60,
        final_k: int = 20,
        primary_query_weight: float = 1.0,
        candidate_k: Optional[int] = 50,
    ) -> List[RetrievedChunk]:
        """Execute the full search pipeline and return aggregated, ranked chunks.

        The aggregation uses Reciprocal Rank Fusion (RRF) across the multiple
        queries: for each query's results, a chunk receives `weight * (1 / (k + rank))`.

        Parameters
        ----------
        query : str
            The raw user query.
        max_variants : int
            Maximum number of reformulations to generate.
        vector_match_count : int
            Candidates requested from the vector retriever per query.
        keyword_match_count : int
            Candidates requested from the keyword retriever per query.
        filter_domain : Optional[str]
            Optional domain filter.
        k : int
            RRF damping factor for multi-query aggregation.
        final_k : int
            Number of top results to return overall.
        primary_query_weight : float
            Weight multiplier for the normalized original user query branch.
        candidate_k : Optional[int]
            Depth of each query's fused hybrid list fed into multi-query RRF.
            Deeper lists give RRF more evidence per chunk (Phase 5: top-10
            recall 0.83 -> 0.87 going from 20 to 50). ``None`` uses ``final_k``.
        """
        return self.search_detailed(
            query,
            max_variants=max_variants,
            vector_match_count=vector_match_count,
            keyword_match_count=keyword_match_count,
            filter_domain=filter_domain,
            k=k,
            final_k=final_k,
            primary_query_weight=primary_query_weight,
            candidate_k=candidate_k,
        ).results

    def search_detailed(
        self,
        query: str,
        max_variants: int = 5,
        vector_match_count: int = 20,
        keyword_match_count: int = 20,
        filter_domain: Optional[str] = None,
        k: int = 60,
        final_k: int = 20,
        primary_query_weight: float = 1.0,
        candidate_k: Optional[int] = 50,
    ) -> SearchResult:
        """Same as :meth:`search`, but returns a :class:`SearchResult` with a trace."""
        timings: Dict[str, float] = {}
        t0 = time.perf_counter()

        # 1 & 2. Validation and Normalization
        validate_query(query)
        normalized = normalize_query(query)

        # 3. Reformulation
        t = time.perf_counter()
        reformulations: List[str] = []
        reformulation_error: Optional[str] = None
        cache_key = (query.strip(), max_variants)
        cached = self._reformulation_cache.get(cache_key) if self.reformulation_cache_size else None
        if cached is not None:
            self._reformulation_cache.move_to_end(cache_key)
            reformulations = list(cached)
        else:
            try:
                reformulations = self.reformulator.generate_variants(
                    query, max_variants=max_variants
                )
                self._remember_reformulations(cache_key, reformulations)
            except Exception as exc:
                # Fall back gracefully to just the original query on LLM/reformulation errors
                logger.warning("Query reformulation failed, falling back to original query: %s", exc)
                reformulation_error = str(exc)
                reformulations = []
        timings["reformulation"] = (time.perf_counter() - t) * 1000

        # 4. Construct candidate queries list (Original + Reformulations)
        candidate_queries = [normalized] + reformulations

        # 5. Semantic Similarity Filtering
        t = time.perf_counter()
        try:
            final_queries = self.semantic_similarity.filter_redundant(candidate_queries)
        except Exception as exc:
            logger.warning("Semantic similarity filtering failed, using unfiltered queries: %s", exc)
            final_queries = candidate_queries

        # If somehow filtering drops the original query and results in empty list, safeguard it.
        if not final_queries:
            final_queries = [normalized]
        timings["filtering"] = (time.perf_counter() - t) * 1000

        # 6 & 7 & 8. Hybrid Retrieval, Aggregation and Ranking
        t = time.perf_counter()
        results = self.aggregate_retrieval(
            queries=final_queries,
            vector_match_count=vector_match_count,
            keyword_match_count=keyword_match_count,
            filter_domain=filter_domain,
            k=k,
            final_k=final_k,
            primary_query_weight=primary_query_weight,
            candidate_k=candidate_k,
        )
        timings["retrieval"] = (time.perf_counter() - t) * 1000
        timings["total"] = (time.perf_counter() - t0) * 1000

        return SearchResult(
            results=results,
            original_query=query,
            normalized_query=normalized,
            reformulations=list(reformulations),
            queries_used=list(final_queries),
            timings_ms={name: round(ms, 1) for name, ms in timings.items()},
            reformulation_error=reformulation_error,
            reformulation_cached=cached is not None,
        )

    def _remember_reformulations(self, key: tuple[str, int], reformulations: List[str]) -> None:
        if not self.reformulation_cache_size:
            return
        self._reformulation_cache[key] = list(reformulations)
        self._reformulation_cache.move_to_end(key)
        while len(self._reformulation_cache) > self.reformulation_cache_size:
            self._reformulation_cache.popitem(last=False)

    def aggregate_retrieval(
        self,
        queries: List[str],
        vector_match_count: int = 20,
        keyword_match_count: int = 20,
        filter_domain: Optional[str] = None,
        k: int = 60,
        final_k: int = 20,
        primary_query_weight: float = 1.0,
        candidate_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Execute hybrid retrieval across multiple queries and aggregate using RRF.

        Queries are retrieved concurrently (up to ``max_workers``), but fused in
        their original order, so results are identical to sequential execution.

        Parameters
        ----------
        queries : List[str]
            List of search queries to retrieve and aggregate.
        vector_match_count : int
            Candidates requested from the vector retriever per query.
        keyword_match_count : int
            Candidates requested from the keyword retriever per query.
        filter_domain : Optional[str]
            Optional domain filter.
        k : int
            RRF damping factor for multi-query aggregation.
        final_k : int
            Number of top results to return overall.
        primary_query_weight : float
            Weight multiplier for the first query in queries.
        candidate_k : Optional[int]
            Depth of each query's hybrid result list (``None`` = ``final_k``).
        """
        per_query_k = candidate_k or final_k

        def retrieve_one(q: str) -> List[RetrievedChunk]:
            try:
                return self.hybrid_retriever.retrieve(
                    query=q,
                    vector_match_count=vector_match_count,
                    keyword_match_count=keyword_match_count,
                    filter_domain=filter_domain,
                    k=k,
                    final_k=per_query_k,
                )
            except Exception as exc:
                logger.error("Hybrid retrieval failed for query '%s': %s", q, exc)
                return []

        if self.max_workers > 1 and len(queries) > 1:
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(queries))) as pool:
                per_query_results = list(pool.map(retrieve_one, queries))
        else:
            per_query_results = [retrieve_one(q) for q in queries]

        fused: dict[str, RetrievedChunk] = {}

        for q_idx, results in enumerate(per_query_results):
            weight = primary_query_weight if q_idx == 0 else 1.0

            for rank, chunk in enumerate(results, start=1):
                contribution = weight * (1.0 / (k + rank))

                if chunk.chunk_id not in fused:
                    new_chunk = dataclasses.replace(chunk, score=0.0, metadata=dict(chunk.metadata))
                    fused[chunk.chunk_id] = new_chunk

                fused[chunk.chunk_id].score += contribution

                if chunk.metadata:
                    fused[chunk.chunk_id].metadata.update(chunk.metadata)

        aggregated_results = list(fused.values())
        aggregated_results.sort(key=lambda c: (-c.score, c.chunk_id))

        return aggregated_results[:final_k]
