import unittest
from unittest.mock import MagicMock, patch
from typing import List

from retrieval.models import RetrievedChunk
from retrieval.hybrid_retriever import HybridRetriever


def make_chunk(chunk_id: str, document_id: str = "doc1", title: str = "Test Doc", domain: str = "TestDomain",
               chunk_index: int = 0, chunk_text: str = "Sample text", score: float = 0.0,
               retriever_type: str = "vector", metadata: dict = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        title=title,
        domain=domain,
        chunk_index=chunk_index,
        chunk_text=chunk_text,
        section=None,
        token_count=None,
        score=score,
        metadata=metadata or {},
        retriever_type=retriever_type,
    )


class TestHybridRetriever(unittest.TestCase):
    def setUp(self):
        # Create mocks for the two retrievers
        self.vec_mock = MagicMock()
        self.key_mock = MagicMock()
        self.hybrid = HybridRetriever(vector_retriever=self.vec_mock, keyword_retriever=self.key_mock)
        self.query = "test query"
        self.filter_domain = "Academic"
        self.k = 60

    def test_overlapping_results(self):
        # Vector returns c1 rank1, c2 rank2
        vec_chunks = [make_chunk("c1", score=0.9, retriever_type="vector"),
                      make_chunk("c2", score=0.8, retriever_type="vector")]
        # Keyword returns c1 rank2, c3 rank1
        key_chunks = [make_chunk("c1", score=0.7, retriever_type="keyword"),
                      make_chunk("c3", score=0.85, retriever_type="keyword")]
        self.vec_mock.retrieve.return_value = vec_chunks
        self.key_mock.retrieve.return_value = key_chunks

        result = self.hybrid.retrieve(self.query, vector_match_count=2, keyword_match_count=2,
                                      filter_domain=self.filter_domain, k=self.k, final_k=3)
        # Expected RRF scores
        rrf_c1 = 1 / (self.k + 1) + 1 / (self.k + 2)
        rrf_c2 = 1 / (self.k + 2)
        rrf_c3 = 1 / (self.k + 1)
        # Verify ordering by RRF (c1 highest, then c3, then c2)
        self.assertEqual([c.chunk_id for c in result], ["c1", "c3", "c2"])
        # Verify scores attached to objects
        scores = {c.chunk_id: c.score for c in result}
        self.assertAlmostEqual(scores["c1"], rrf_c1)
        self.assertAlmostEqual(scores["c2"], rrf_c2)
        self.assertAlmostEqual(scores["c3"], rrf_c3)
        # Ensure original vector/keyword scores are still accessible via metadata for tie‑breakers
        # (they are stored in the object attributes before being overwritten with RRF)
        self.assertEqual(vec_chunks[0].score, 0.9)
        self.assertEqual(key_chunks[0].score, 0.7)

    def test_vector_only(self):
        vec_chunks = [make_chunk("c1", score=0.9, retriever_type="vector")]
        self.vec_mock.retrieve.return_value = vec_chunks
        self.key_mock.retrieve.return_value = []
        result = self.hybrid.retrieve(self.query, vector_match_count=1, keyword_match_count=0,
                                      filter_domain=self.filter_domain, k=self.k, final_k=5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].chunk_id, "c1")
        # RRF for single vector entry should be 1/(k+1)
        self.assertAlmostEqual(result[0].score, 1 / (self.k + 1))

    def test_keyword_only(self):
        key_chunks = [make_chunk("c2", score=0.85, retriever_type="keyword")]
        self.vec_mock.retrieve.return_value = []
        self.key_mock.retrieve.return_value = key_chunks
        result = self.hybrid.retrieve(self.query, vector_match_count=0, keyword_match_count=1,
                                      filter_domain=self.filter_domain, k=self.k, final_k=5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].chunk_id, "c2")
        self.assertAlmostEqual(result[0].score, 1 / (self.k + 1))

    def test_duplicate_chunk_single_output(self):
        # Same chunk_id in both lists
        vec_chunks = [make_chunk("c1", score=0.9, retriever_type="vector")]
        key_chunks = [make_chunk("c1", score=0.75, retriever_type="keyword")]
        self.vec_mock.retrieve.return_value = vec_chunks
        self.key_mock.retrieve.return_value = key_chunks
        result = self.hybrid.retrieve(self.query, vector_match_count=1, keyword_match_count=1,
                                      filter_domain=self.filter_domain, k=self.k, final_k=5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].chunk_id, "c1")
        # RRF should be sum of both ranks (both rank 1)
        expected = 1 / (self.k + 1) + 1 / (self.k + 1)
        self.assertAlmostEqual(result[0].score, expected)

    def test_rank_starts_at_one(self):
        vec_chunks = [make_chunk(f"v{i}", score=0.5) for i in range(1, 3)]
        self.vec_mock.retrieve.return_value = vec_chunks
        self.key_mock.retrieve.return_value = []
        result = self.hybrid.retrieve(self.query, vector_match_count=2, keyword_match_count=0,
                                      filter_domain=self.filter_domain, k=self.k, final_k=2)
        # First result should have rank 1 -> 1/(k+1)
        self.assertAlmostEqual(result[0].score, 1 / (self.k + 1))
        # Second result rank 2 -> 1/(k+2)
        self.assertAlmostEqual(result[1].score, 1 / (self.k + 2))

    def test_final_k_truncation(self):
        vec_chunks = [make_chunk(f"c{i}", score=0.5) for i in range(1, 6)]
        self.vec_mock.retrieve.return_value = vec_chunks
        self.key_mock.retrieve.return_value = []
        result = self.hybrid.retrieve(self.query, vector_match_count=5, keyword_match_count=0,
                                      filter_domain=self.filter_domain, k=self.k, final_k=3)
        self.assertEqual(len(result), 3)
        self.assertEqual([c.chunk_id for c in result], ["c1", "c2", "c3"])

    def test_deterministic_ordering(self):
        vec_chunks = [make_chunk("c1", score=0.9), make_chunk("c2", score=0.8)]
        key_chunks = [make_chunk("c2", score=0.85), make_chunk("c1", score=0.7)]
        self.vec_mock.retrieve.return_value = vec_chunks
        self.key_mock.retrieve.return_value = key_chunks
        first = self.hybrid.retrieve(self.query, vector_match_count=2, keyword_match_count=2,
                                      filter_domain=self.filter_domain, k=self.k, final_k=2)
        second = self.hybrid.retrieve(self.query, vector_match_count=2, keyword_match_count=2,
                                      filter_domain=self.filter_domain, k=self.k, final_k=2)
        self.assertEqual([c.chunk_id for c in first], [c.chunk_id for c in second])

    def test_domain_filter_passed(self):
        self.vec_mock.retrieve.return_value = []
        self.key_mock.retrieve.return_value = []
        self.hybrid.retrieve(self.query, vector_match_count=3, keyword_match_count=4,
                             filter_domain=self.filter_domain, k=self.k, final_k=5)
        self.vec_mock.retrieve.assert_called_once_with(query=self.query, match_count=3, filter_domain=self.filter_domain)
        self.key_mock.retrieve.assert_called_once_with(query=self.query, match_count=4, filter_domain=self.filter_domain)

    def test_zero_match_count_skips_branch(self):
        self.vec_mock.retrieve.return_value = [make_chunk("c1", score=0.9)]
        result = self.hybrid.retrieve(self.query, vector_match_count=2, keyword_match_count=0,
                                      filter_domain=self.filter_domain, k=self.k, final_k=5)
        self.key_mock.retrieve.assert_not_called()
        self.assertEqual([c.chunk_id for c in result], ["c1"])

    def test_both_empty(self):
        self.vec_mock.retrieve.return_value = []
        self.key_mock.retrieve.return_value = []
        result = self.hybrid.retrieve(self.query, vector_match_count=0, keyword_match_count=0,
                                      filter_domain=self.filter_domain, k=self.k, final_k=5)
        self.assertEqual(result, [])
        self.vec_mock.retrieve.assert_not_called()
        self.key_mock.retrieve.assert_not_called()

    def test_configurable_k(self):
        vec_chunks = [make_chunk("c1", score=0.9)]
        key_chunks = [make_chunk("c1", score=0.8)]
        self.vec_mock.retrieve.return_value = vec_chunks
        self.key_mock.retrieve.return_value = key_chunks
        # k=10
        result_k10 = self.hybrid.retrieve(self.query, vector_match_count=1, keyword_match_count=1,
                                          filter_domain=self.filter_domain, k=10, final_k=5)
        expected_k10 = 1 / (10 + 1) + 1 / (10 + 1)
        self.assertAlmostEqual(result_k10[0].score, expected_k10)
        # k=60
        result_k60 = self.hybrid.retrieve(self.query, vector_match_count=1, keyword_match_count=1,
                                          filter_domain=self.filter_domain, k=60, final_k=5)
        expected_k60 = 1 / (60 + 1) + 1 / (60 + 1)
        self.assertAlmostEqual(result_k60[0].score, expected_k60)
        self.assertNotEqual(result_k10[0].score, result_k60[0].score)

    def test_one_branch_failure(self):
        # Vector raises, keyword succeeds
        self.vec_mock.retrieve.side_effect = RuntimeError("Vector down")
        key_chunks = [make_chunk("c1", score=0.8, retriever_type="keyword")]
        self.key_mock.retrieve.return_value = key_chunks
        result = self.hybrid.retrieve(self.query, vector_match_count=1, keyword_match_count=1,
                                      filter_domain=self.filter_domain, k=self.k, final_k=5)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].chunk_id, "c1")
        # Score should be 1/(k+1) from keyword rank 1
        self.assertAlmostEqual(result[0].score, 1 / (self.k + 1))

    def test_both_branches_failure(self):
        self.vec_mock.retrieve.side_effect = RuntimeError("Vector down")
        self.key_mock.retrieve.side_effect = RuntimeError("Keyword down")
        with self.assertRaises(RuntimeError) as ctx:
            self.hybrid.retrieve(self.query, vector_match_count=1, keyword_match_count=1,
                                 filter_domain=self.filter_domain, k=self.k, final_k=5)
        self.assertIn("Both retrieval branches failed", str(ctx.exception))

    def test_branch_failure_reported_to_callback(self):
        self.key_mock.retrieve.side_effect = RuntimeError("Keyword down")
        self.vec_mock.retrieve.return_value = [make_chunk("c1", score=0.9)]
        failures = []
        result = self.hybrid.retrieve(self.query, vector_match_count=1, keyword_match_count=1,
                                      on_branch_error=lambda branch, exc: failures.append((branch, str(exc))))
        self.assertEqual([c.chunk_id for c in result], ["c1"])
        self.assertEqual(failures, [("keyword", "Keyword down")])

    def test_callback_not_called_when_healthy(self):
        self.vec_mock.retrieve.return_value = [make_chunk("c1", score=0.9)]
        self.key_mock.retrieve.return_value = []
        failures = []
        self.hybrid.retrieve(self.query, on_branch_error=lambda branch, exc: failures.append(branch))
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
