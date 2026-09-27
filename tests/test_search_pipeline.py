import unittest
from unittest.mock import MagicMock
import dataclasses

from reformulation.search_pipeline import SearchPipeline
from reformulation.reformulator import QueryReformulator
from reformulation.semantic_similarity import SemanticSimilarity
from reformulation.normalizer import normalize_query
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import RetrievedChunk


class TestSearchPipeline(unittest.TestCase):
    def setUp(self):
        # Mocks
        self.reformulator = MagicMock(spec=QueryReformulator)
        self.semantic_similarity = MagicMock(spec=SemanticSimilarity)
        self.hybrid_retriever = MagicMock(spec=HybridRetriever)

        # Pipeline instance
        self.pipeline = SearchPipeline(
            reformulator=self.reformulator,
            semantic_similarity=self.semantic_similarity,
            hybrid_retriever=self.hybrid_retriever,
        )

        # Default behaviors
        self.reformulator.generate_variants.return_value = ["variant 1", "variant 2"]
        # By default, pass through
        self.semantic_similarity.filter_redundant.side_effect = lambda qs: qs

        # The normalized form of the raw input "original query" used in these tests.
        # normalize_query applies sentence-case, so "original query" → "Original query".
        self._norm_original = normalize_query("original query")

        def mock_retrieve(query, **kwargs):
            if query == self._norm_original:
                return [
                    RetrievedChunk(chunk_id="A", document_id="d1", title="T", domain="D", chunk_index=0, chunk_text="A", section=None, token_count=10, score=1.0),
                    RetrievedChunk(chunk_id="B", document_id="d2", title="T", domain="D", chunk_index=0, chunk_text="B", section=None, token_count=10, score=0.9),
                ]
            elif query == "variant 1":
                return [
                    RetrievedChunk(chunk_id="B", document_id="d2", title="T", domain="D", chunk_index=0, chunk_text="B", section=None, token_count=10, score=0.95),
                    RetrievedChunk(chunk_id="C", document_id="d3", title="T", domain="D", chunk_index=0, chunk_text="C", section=None, token_count=10, score=0.8),
                ]
            else:
                return []
        
        self.hybrid_retriever.retrieve.side_effect = mock_retrieve

    def test_basic_end_to_end_pipeline(self):
        results = self.pipeline.search("original query")
        
        # Verify validation and normalization happened implicitly.
        # The reformulator is called with the RAW query (as per pipeline design).
        self.reformulator.generate_variants.assert_called_once_with("original query", max_variants=5)
        # The semantic similarity filter receives the NORMALIZED query first, then variants.
        self.semantic_similarity.filter_redundant.assert_called_once_with(
            [self._norm_original, "variant 1", "variant 2"]
        )
        
        # Ensure hybrid retriever was called for the original query and the variants
        self.assertEqual(self.hybrid_retriever.retrieve.call_count, 3)
        
        # We expect chunks A, B, C
        chunk_ids = [c.chunk_id for c in results]
        self.assertCountEqual(chunk_ids, ["A", "B", "C"])

    def test_aggregation_and_rank_logic(self):
        # We enforce a specific rank test
        # Query 1 -> rank 1: A, rank 2: B, rank 3: C
        # Query 2 -> rank 1: B, rank 2: C, rank 3: D

        # Use already-normalised, clean query strings so normalization is a no-op.
        norm_q1 = normalize_query("Q1")
        
        def mock_retrieve_agg(query, **kwargs):
            if query == norm_q1:
                return [
                    RetrievedChunk(chunk_id="A", document_id="d", title="t", domain="d", chunk_index=0, chunk_text="t", section=None, token_count=1, score=1.0),
                    RetrievedChunk(chunk_id="B", document_id="d", title="t", domain="d", chunk_index=0, chunk_text="t", section=None, token_count=1, score=0.9),
                    RetrievedChunk(chunk_id="C", document_id="d", title="t", domain="d", chunk_index=0, chunk_text="t", section=None, token_count=1, score=0.8),
                ]
            elif query == "Q2":
                return [
                    RetrievedChunk(chunk_id="B", document_id="d", title="t", domain="d", chunk_index=0, chunk_text="t", section=None, token_count=1, score=1.0),
                    RetrievedChunk(chunk_id="C", document_id="d", title="t", domain="d", chunk_index=0, chunk_text="t", section=None, token_count=1, score=0.9),
                    RetrievedChunk(chunk_id="D", document_id="d", title="t", domain="d", chunk_index=0, chunk_text="t", section=None, token_count=1, score=0.8),
                ]
            return []
            
        self.hybrid_retriever.retrieve.side_effect = mock_retrieve_agg
        self.reformulator.generate_variants.return_value = ["Q2"]
        self.semantic_similarity.filter_redundant.side_effect = lambda qs: qs
        
        results = self.pipeline.search("Q1", k=60)
        
        # Calculate expected scores
        # A: 1/(60+1) from Q1
        # B: 1/(60+2) from Q1 + 1/(60+1) from Q2
        # C: 1/(60+3) from Q1 + 1/(60+2) from Q2
        # D: 1/(60+3) from Q2
        score_A = 1.0 / 61
        score_B = 1.0 / 62 + 1.0 / 61
        score_C = 1.0 / 63 + 1.0 / 62
        score_D = 1.0 / 63
        
        # Expected order: B (highest), C, A, D
        self.assertEqual(len(results), 4)
        self.assertEqual(results[0].chunk_id, "B")
        self.assertEqual(results[1].chunk_id, "C")
        self.assertEqual(results[2].chunk_id, "A")
        self.assertEqual(results[3].chunk_id, "D")
        
        self.assertAlmostEqual(results[0].score, score_B)
        self.assertAlmostEqual(results[1].score, score_C)
        self.assertAlmostEqual(results[2].score, score_A)
        self.assertAlmostEqual(results[3].score, score_D)

    def test_final_k_is_respected(self):
        self.hybrid_retriever.retrieve.side_effect = None
        self.hybrid_retriever.retrieve.return_value = [
            RetrievedChunk(chunk_id=str(i), document_id="d", title="t", domain="d", chunk_index=0, chunk_text="t", section=None, token_count=1, score=1.0)
            for i in range(10)
        ]
        self.reformulator.generate_variants.return_value = []
        
        results = self.pipeline.search("query", final_k=3)
        self.assertEqual(len(results), 3)

    def test_deterministic_ordering_tie_breaker(self):
        # Q1 and Q2 produce Z and A at same rank, so their scores tie.
        # chunk_id string comparison will break the tie.
        norm_q1 = normalize_query("Q1")

        def mock_retrieve_tie(query, **kwargs):
            if query == norm_q1:
                return [
                    RetrievedChunk(chunk_id="Z", document_id="d", title="t", domain="d", chunk_index=0, chunk_text="t", section=None, token_count=1, score=1.0)
                ]
            elif query == "Q2":
                return [
                    RetrievedChunk(chunk_id="A", document_id="d", title="t", domain="d", chunk_index=0, chunk_text="t", section=None, token_count=1, score=1.0)
                ]
            return []
            
        self.hybrid_retriever.retrieve.side_effect = mock_retrieve_tie
        self.reformulator.generate_variants.return_value = ["Q2"]
        
        results = self.pipeline.search("Q1")
        self.assertEqual(results[0].chunk_id, "A")
        self.assertEqual(results[1].chunk_id, "Z")

    def test_no_reformulations_searches_original_query(self):
        self.hybrid_retriever.retrieve.side_effect = None
        self.reformulator.generate_variants.return_value = []
        self.hybrid_retriever.retrieve.return_value = [
            RetrievedChunk(chunk_id="A", document_id="d", title="t", domain="d", chunk_index=0, chunk_text="t", section=None, token_count=1, score=1.0)
        ]
        
        results = self.pipeline.search("query")
        
        # The normalized form of "query" is "Query" (sentence case)
        expected_normalized = normalize_query("query")
        self.hybrid_retriever.retrieve.assert_called_once_with(
            query=expected_normalized, vector_match_count=20, keyword_match_count=20, filter_domain=None, k=60, final_k=50
        )
        self.assertEqual(len(results), 1)

    def test_reformulator_failure_fallback(self):
        self.hybrid_retriever.retrieve.side_effect = None
        self.reformulator.generate_variants.side_effect = Exception("LLM Error")
        self.hybrid_retriever.retrieve.return_value = [
            RetrievedChunk(chunk_id="A", document_id="d", title="t", domain="d", chunk_index=0, chunk_text="t", section=None, token_count=1, score=1.0)
        ]
        
        # Pipeline should catch and fallback to normalized query
        results = self.pipeline.search("query")
        
        expected_normalized = normalize_query("query")
        self.hybrid_retriever.retrieve.assert_called_once_with(
            query=expected_normalized, vector_match_count=20, keyword_match_count=20, filter_domain=None, k=60, final_k=50
        )
        self.assertEqual(len(results), 1)

    def test_retrieval_branch_failure_continues(self):
        self.reformulator.generate_variants.return_value = ["bad variant"]
        
        norm_query = normalize_query("query")

        def mock_retrieve_fail(query, **kwargs):
            if query == norm_query:
                return [RetrievedChunk(chunk_id="A", document_id="d", title="t", domain="d", chunk_index=0, chunk_text="t", section=None, token_count=1, score=1.0)]
            else:
                raise RuntimeError("Retrieval failed")
                
        self.hybrid_retriever.retrieve.side_effect = mock_retrieve_fail
        
        # Should catch error for 'bad variant' but succeed overall with 'query' results
        results = self.pipeline.search("query")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk_id, "A")

    def test_empty_retrieval_results(self):
        self.hybrid_retriever.retrieve.side_effect = None
        self.hybrid_retriever.retrieve.return_value = []
        results = self.pipeline.search("query")
        self.assertEqual(results, [])

    def test_metadata_preserved(self):
        self.hybrid_retriever.retrieve.side_effect = None
        self.hybrid_retriever.retrieve.return_value = [
            RetrievedChunk(chunk_id="A", document_id="doc1", title="T", domain="D", chunk_index=5, chunk_text="text", section="sec", token_count=100, score=1.0, metadata={"source": "test"})
        ]
        results = self.pipeline.search("query")
        
        chunk = results[0]
        self.assertEqual(chunk.chunk_id, "A")
        self.assertEqual(chunk.document_id, "doc1")
        self.assertEqual(chunk.title, "T")
        self.assertEqual(chunk.domain, "D")
        self.assertEqual(chunk.chunk_index, 5)
        self.assertEqual(chunk.chunk_text, "text")
        self.assertEqual(chunk.section, "sec")
        self.assertEqual(chunk.token_count, 100)
        self.assertEqual(chunk.metadata["source"], "test")

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            self.pipeline.search("")
        with self.assertRaises(ValueError):
            self.pipeline.search(None)

    def test_primary_query_weight_mathematical_behavior(self):
        # Test that primary query branch receives primary_query_weight multiplier
        # while reformulation branches receive 1.0 multiplier.
        norm_q = normalize_query("main query")
        
        def mock_retrieve_weighted(query, **kwargs):
            if query == norm_q:
                # Rank 1: ChunkA, Rank 2: ChunkB
                return [
                    RetrievedChunk(chunk_id="ChunkA", document_id="d1", title="T", domain="D", chunk_index=0, chunk_text="A", section=None, token_count=10, score=1.0),
                    RetrievedChunk(chunk_id="ChunkB", document_id="d2", title="T", domain="D", chunk_index=0, chunk_text="B", section=None, token_count=10, score=0.9),
                ]
            elif query == "variant 1":
                # Rank 1: ChunkB, Rank 2: ChunkC
                return [
                    RetrievedChunk(chunk_id="ChunkB", document_id="d2", title="T", domain="D", chunk_index=0, chunk_text="B", section=None, token_count=10, score=1.0),
                    RetrievedChunk(chunk_id="ChunkC", document_id="d3", title="T", domain="D", chunk_index=0, chunk_text="C", section=None, token_count=10, score=0.8),
                ]
            return []

        self.hybrid_retriever.retrieve.side_effect = mock_retrieve_weighted
        self.reformulator.generate_variants.return_value = ["variant 1"]
        self.semantic_similarity.filter_redundant.side_effect = lambda qs: qs

        # 1. Unweighted / Baseline (weight = 1.0)
        # Score ChunkA = 1.0 * (1 / (60 + 1)) = 1/61 = 0.01639344
        # Score ChunkB = 1.0 * (1 / (60 + 2)) + 1.0 * (1 / (60 + 1)) = 1/62 + 1/61 = 0.03252247
        # Score ChunkC = 1.0 * (1 / (60 + 2)) = 1/62 = 0.01612903
        results_1_0 = self.pipeline.search("main query", k=60, primary_query_weight=1.0)
        scores_1_0 = {c.chunk_id: c.score for c in results_1_0}
        self.assertAlmostEqual(scores_1_0["ChunkA"], 1.0 / 61)
        self.assertAlmostEqual(scores_1_0["ChunkB"], 1.0 / 62 + 1.0 / 61)
        self.assertAlmostEqual(scores_1_0["ChunkC"], 1.0 / 62)
        self.assertEqual(results_1_0[0].chunk_id, "ChunkB")
        self.assertEqual(results_1_0[1].chunk_id, "ChunkA")
        self.assertEqual(results_1_0[2].chunk_id, "ChunkC")

        # 2. Weighted (weight = 1.5)
        # Score ChunkA = 1.5 * (1 / 61) = 0.02459016
        # Score ChunkB = 1.5 * (1 / 62) + 1.0 * (1 / 61) = 0.02419355 + 0.01639344 = 0.04058699
        # Score ChunkC = 1.0 * (1 / 62) = 0.01612903
        results_1_5 = self.pipeline.search("main query", k=60, primary_query_weight=1.5)
        scores_1_5 = {c.chunk_id: c.score for c in results_1_5}
        self.assertAlmostEqual(scores_1_5["ChunkA"], 1.5 / 61)
        self.assertAlmostEqual(scores_1_5["ChunkB"], 1.5 / 62 + 1.0 / 61)
        self.assertAlmostEqual(scores_1_5["ChunkC"], 1.0 / 62)

        # 3. Default parameter compatibility (omitted primary_query_weight defaults to 1.0)
        results_default = self.pipeline.search("main query", k=60)
        scores_default = {c.chunk_id: c.score for c in results_default}
        self.assertEqual(scores_default, scores_1_0)


class TestSearchPipelineRuntime(unittest.TestCase):
    """Candidate depth, parallel retrieval, reformulation cache and trace."""

    def _chunk(self, cid, score=1.0, metadata=None):
        return RetrievedChunk(chunk_id=cid, document_id="d", title="T", domain="D", chunk_index=0,
                              chunk_text=cid, section=None, token_count=1, score=score, metadata=metadata or {})

    def _pipeline(self, **kwargs):
        reformulator = MagicMock(spec=QueryReformulator)
        reformulator.generate_variants.return_value = ["v1", "v2", "v3"]
        sem = MagicMock(spec=SemanticSimilarity)
        sem.filter_redundant.side_effect = lambda qs: qs
        hybrid = MagicMock(spec=HybridRetriever)
        return SearchPipeline(reformulator=reformulator, semantic_similarity=sem, hybrid_retriever=hybrid, **kwargs)

    def test_candidate_k_sets_per_query_depth(self):
        p = self._pipeline()
        p.hybrid_retriever.retrieve.return_value = []
        p.search("query", final_k=10, candidate_k=40)
        for call in p.hybrid_retriever.retrieve.call_args_list:
            self.assertEqual(call.kwargs["final_k"], 40)
        p.hybrid_retriever.retrieve.reset_mock()
        p.search("query", final_k=10, candidate_k=None)
        for call in p.hybrid_retriever.retrieve.call_args_list:
            self.assertEqual(call.kwargs["final_k"], 10)

    def test_parallel_matches_sequential(self):
        lists = {
            normalize_query("query"): ["A", "B", "C"],
            "v1": ["C", "D"],
            "v2": ["B", "E", "A"],
            "v3": ["F"],
        }
        outputs = []
        for workers in (1, 4):
            p = self._pipeline(max_workers=workers)
            p.hybrid_retriever.retrieve.side_effect = lambda query, **kw: [self._chunk(c) for c in lists[query]]
            outputs.append([(c.chunk_id, round(c.score, 12)) for c in p.search("query", primary_query_weight=2.0)])
        self.assertEqual(outputs[0], outputs[1])

    def test_reformulations_cached_for_repeated_query(self):
        p = self._pipeline()
        p.hybrid_retriever.retrieve.return_value = []
        first = p.search_detailed("query")
        second = p.search_detailed("query")
        p.reformulator.generate_variants.assert_called_once()
        self.assertFalse(first.reformulation_cached)
        self.assertTrue(second.reformulation_cached)
        self.assertEqual(first.reformulations, second.reformulations)

    def test_cache_disabled_and_eviction(self):
        p = self._pipeline(reformulation_cache_size=0)
        p.hybrid_retriever.retrieve.return_value = []
        p.search("query"); p.search("query")
        self.assertEqual(p.reformulator.generate_variants.call_count, 2)

        p = self._pipeline(reformulation_cache_size=1)
        p.hybrid_retriever.retrieve.return_value = []
        p.search("query a"); p.search("query b"); p.search("query a")
        self.assertEqual(p.reformulator.generate_variants.call_count, 3)

    def test_failed_reformulation_not_cached(self):
        p = self._pipeline()
        p.hybrid_retriever.retrieve.return_value = []
        p.reformulator.generate_variants.side_effect = [RuntimeError("LLM down"), ["v1"]]
        first = p.search_detailed("query")
        second = p.search_detailed("query")
        self.assertEqual(first.reformulation_error, "LLM down")
        self.assertEqual(second.reformulations, ["v1"])

    def test_search_detailed_trace(self):
        p = self._pipeline()
        p.hybrid_retriever.retrieve.return_value = [self._chunk("A")]
        res = p.search_detailed("original query")
        self.assertEqual(res.original_query, "original query")
        self.assertEqual(res.normalized_query, normalize_query("original query"))
        self.assertEqual(res.queries_used, [normalize_query("original query"), "v1", "v2", "v3"])
        self.assertEqual([c.chunk_id for c in res.results], ["A"])
        self.assertEqual(set(res.timings_ms), {"reformulation", "filtering", "retrieval", "total"})
        self.assertEqual(res.to_dict()["results"][0]["chunk_id"], "A")

    def test_retriever_chunks_not_mutated(self):
        p = self._pipeline()
        shared = self._chunk("A", metadata={"x": 1})
        other = self._chunk("A", metadata={"y": 2})
        p.hybrid_retriever.retrieve.side_effect = lambda query, **kw: [shared] if query == "v1" else [other]
        p.search("query")
        self.assertEqual(shared.metadata, {"x": 1})
        self.assertEqual(other.metadata, {"y": 2})
        self.assertEqual(shared.score, 1.0)

    def test_invalid_constructor_args(self):
        with self.assertRaises(ValueError):
            self._pipeline(max_workers=0)
        with self.assertRaises(ValueError):
            self._pipeline(reformulation_cache_size=-1)


if __name__ == "__main__":
    unittest.main()
