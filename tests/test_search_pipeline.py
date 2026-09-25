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
            query=expected_normalized, vector_match_count=20, keyword_match_count=20, filter_domain=None, k=60, final_k=20
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
            query=expected_normalized, vector_match_count=20, keyword_match_count=20, filter_domain=None, k=60, final_k=20
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

    def test_semantic_filtering_failure_fallback(self):
        self.semantic_similarity.filter_redundant.side_effect = Exception("Semantic Error")
        self.reformulator.generate_variants.return_value = ["v1"]
        self.hybrid_retriever.retrieve.return_value = []
        
        # Pipeline should catch and fallback to the unfiltered candidate list
        self.pipeline.search("q")
        
        self.assertEqual(self.hybrid_retriever.retrieve.call_count, 2) # original + v1

if __name__ == "__main__":
    unittest.main()
