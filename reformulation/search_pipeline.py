import logging
import dataclasses
from typing import List, Optional

from reformulation.validator import validate_query
from reformulation.normalizer import normalize_query
from reformulation.reformulator import QueryReformulator
from reformulation.semantic_similarity import SemanticSimilarity
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


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
        """
        self.reformulator = reformulator
        self.semantic_similarity = semantic_similarity
        self.hybrid_retriever = hybrid_retriever

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
        """
        # 1 & 2. Validation and Normalization
        validate_query(query)
        normalized = normalize_query(query)

        # 3. Reformulation
        reformulations = []
        try:
            reformulations = self.reformulator.generate_variants(
                query, max_variants=max_variants
            )
        except Exception as exc:
            # Fall back gracefully to just the original query on LLM/reformulation errors
            logger.warning("Query reformulation failed, falling back to original query: %s", exc)
            reformulations = []

        # 4. Construct candidate queries list (Original + Reformulations)
        candidate_queries = [normalized] + reformulations

        # 5. Semantic Similarity Filtering
        try:
            final_queries = self.semantic_similarity.filter_redundant(candidate_queries)
        except Exception as exc:
            logger.warning("Semantic similarity filtering failed, using unfiltered queries: %s", exc)
            final_queries = candidate_queries

        # If somehow filtering drops the original query and results in empty list, safeguard it.
        if not final_queries:
            final_queries = [normalized]

        # 6 & 7 & 8. Hybrid Retrieval, Aggregation and Ranking
        return self.aggregate_retrieval(
            queries=final_queries,
            vector_match_count=vector_match_count,
            keyword_match_count=keyword_match_count,
            filter_domain=filter_domain,
            k=k,
            final_k=final_k,
            primary_query_weight=primary_query_weight,
        )

    def aggregate_retrieval(
        self,
        queries: List[str],
        vector_match_count: int = 20,
        keyword_match_count: int = 20,
        filter_domain: Optional[str] = None,
        k: int = 60,
        final_k: int = 20,
        primary_query_weight: float = 1.0,
    ) -> List[RetrievedChunk]:
        """Execute hybrid retrieval across multiple queries and aggregate using RRF.

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
        """
        fused: dict[str, RetrievedChunk] = {}

        for q_idx, q in enumerate(queries):
            weight = primary_query_weight if q_idx == 0 else 1.0
            try:
                results = self.hybrid_retriever.retrieve(
                    query=q,
                    vector_match_count=vector_match_count,
                    keyword_match_count=keyword_match_count,
                    filter_domain=filter_domain,
                    k=k,
                    final_k=final_k,
                )
            except Exception as exc:
                logger.error("Hybrid retrieval failed for query '%s': %s", q, exc)
                continue

            for rank, chunk in enumerate(results, start=1):
                contribution = weight * (1.0 / (k + rank))

                if chunk.chunk_id not in fused:
                    new_chunk = dataclasses.replace(chunk, score=0.0)
                    fused[chunk.chunk_id] = new_chunk

                fused[chunk.chunk_id].score += contribution

                if chunk.metadata:
                    fused[chunk.chunk_id].metadata.update(chunk.metadata)

        aggregated_results = list(fused.values())
        aggregated_results.sort(key=lambda c: (-c.score, c.chunk_id))

        return aggregated_results[:final_k]
