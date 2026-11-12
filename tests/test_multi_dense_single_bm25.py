"""Tests for Phase 2.2 Controlled Multi-Query Dense + Single BM25 Retrieval architecture."""

import unittest
from unittest.mock import MagicMock, call

from reformulation.exceptions import RetrievalUnavailableError
from reformulation.intent_router import IntentRouter, QueryIntent, RoutingAction
from reformulation.normalizer import normalize_query
from reformulation.search_pipeline import SearchPipeline
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.models import RetrievedChunk
from retrieval.vector_retriever import VectorRetriever


class TestMultiDenseSingleBM25(unittest.TestCase):
    """Verify execution behavior of Controlled Multi-Query Dense + Single BM25 retrieval."""

    def _make_chunk(self, chunk_id: str, score: float = 1.0) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id="doc1",
            title="Title",
            domain="Domain",
            chunk_index=0,
            chunk_text=f"Text for {chunk_id}",
            section=None,
            token_count=10,
            score=score,
        )

    def setUp(self):
        self.reformulator = MagicMock()
        self.reformulator.generate_variants.return_value = ["variant 1", "variant 2"]
        self.semantic_similarity = MagicMock()
        self.semantic_similarity.filter_redundant.side_effect = lambda qs, original_query=None: qs

        self.vector_retriever = MagicMock(spec=VectorRetriever)
        self.keyword_retriever = MagicMock(spec=BM25Retriever)
        self.hybrid_retriever = HybridRetriever(
            vector_retriever=self.vector_retriever,
            keyword_retriever=self.keyword_retriever,
        )

        self.router = MagicMock(spec=IntentRouter)
        # Default mock: route to REFORMULATE
        self.router.route.return_value = MagicMock(
            intent=QueryIntent.KEYWORD_LOOKUP,
            action=RoutingAction.REFORMULATE,
            confidence=0.8,
            matched_entity=None,
            reason="Keyword lookup test",
        )

        self.pipeline = SearchPipeline(
            reformulator=self.reformulator,
            semantic_similarity=self.semantic_similarity,
            hybrid_retriever=self.hybrid_retriever,
            intent_router=self.router,
            multi_dense_single_bm25=True,
            fusion_strategy="equal",
            max_workers=1,
        )

    def test_single_bm25_multi_dense_call_counts(self):
        """BM25 should execute once on normalized query; vector search executes on normalized + variants."""
        self.keyword_retriever.retrieve.return_value = [self._make_chunk("chunk_bm25")]
        self.vector_retriever.retrieve.side_effect = [
            [self._make_chunk("chunk_dense_0")],
            [self._make_chunk("chunk_dense_1")],
            [self._make_chunk("chunk_dense_2")],
        ]

        res = self.pipeline.search_detailed("library book borrowing limits")

        # BM25 was executed ONLY ONCE on normalized query
        norm_q = normalize_query("library book borrowing limits")
        self.keyword_retriever.retrieve.assert_called_once_with(
            query=norm_q,
            match_count=20,
            filter_domain=None,
        )

        # Dense retrieval was executed for primary query + 2 variants (3 times total)
        self.assertEqual(self.vector_retriever.retrieve.call_count, 3)
        expected_dense_calls = [
            call(query=norm_q, match_count=20, filter_domain=None),
            call(query="variant 1", match_count=20, filter_domain=None),
            call(query="variant 2", match_count=20, filter_domain=None),
        ]
        self.vector_retriever.retrieve.assert_has_calls(expected_dense_calls)

        # Fused results contain both BM25 and Dense candidates
        chunk_ids = [c.chunk_id for c in res.results]
        self.assertIn("chunk_bm25", chunk_ids)
        self.assertIn("chunk_dense_0", chunk_ids)
        self.assertIn("chunk_dense_1", chunk_ids)
        self.assertIn("chunk_dense_2", chunk_ids)

    def test_weighted_fusion_strategy_applies_multipliers(self):
        """Weighted fusion strategy multiplies primary BM25 and primary Dense contributions."""
        weighted_pipeline = SearchPipeline(
            reformulator=self.reformulator,
            semantic_similarity=self.semantic_similarity,
            hybrid_retriever=self.hybrid_retriever,
            intent_router=self.router,
            multi_dense_single_bm25=True,
            fusion_strategy="weighted",
            max_workers=1,
        )

        norm_q = normalize_query("test query")
        # Setup: chunkA in primary BM25 rank 1
        # chunkB in variant 1 Dense rank 1
        self.keyword_retriever.retrieve.return_value = [self._make_chunk("chunkA")]
        self.vector_retriever.retrieve.side_effect = [
            [],  # primary dense empty
            [self._make_chunk("chunkB")],  # variant 1 dense rank 1
            [],  # variant 2 dense empty
        ]

        res = weighted_pipeline.search_detailed("test query", primary_query_weight=2.0)
        scores = {c.chunk_id: c.score for c in res.results}

        # chunkA should receive weight 2.0 * (1 / (60 + 1)) = 2/61
        # chunkB should receive weight 1.0 * (1 / (60 + 1)) = 1/61
        self.assertAlmostEqual(scores["chunkA"], 2.0 / 61.0)
        self.assertAlmostEqual(scores["chunkB"], 1.0 / 61.0)
        self.assertEqual(res.results[0].chunk_id, "chunkA")
        self.assertEqual(res.results[1].chunk_id, "chunkB")

    def test_bm25_failure_degrades_gracefully(self):
        """If BM25 fails, results continue from the dense branch with a warning."""
        self.keyword_retriever.retrieve.side_effect = RuntimeError("BM25 index corrupted")
        self.vector_retriever.retrieve.return_value = [self._make_chunk("dense_chunk")]

        res = self.pipeline.search_detailed("query")
        self.assertTrue(res.degraded)
        self.assertTrue(any("keyword search branch failed" in w for w in res.warnings))
        self.assertEqual(len(res.results), 1)
        self.assertEqual(res.results[0].chunk_id, "dense_chunk")

    def test_both_branches_failing_raises_exception(self):
        """If both BM25 and all Dense variants fail, raises RetrievalUnavailableError."""
        self.keyword_retriever.retrieve.side_effect = RuntimeError("BM25 down")
        self.vector_retriever.retrieve.side_effect = RuntimeError("Vector DB down")

        with self.assertRaises(RetrievalUnavailableError):
            self.pipeline.search_detailed("query")

    def test_direct_route_preserves_single_hybrid(self):
        """Direct-route queries (e.g. course codes) bypass multi-dense and use single hybrid search."""
        self.router.route.return_value = MagicMock(
            intent=QueryIntent.COURSE_CODE,
            action=RoutingAction.DIRECT,
            confidence=0.98,
            matched_entity="MEC204",
            reason="Course code direct route",
        )
        self.hybrid_retriever.retrieve = MagicMock(return_value=[self._make_chunk("direct_chunk")])

        res = self.pipeline.search_detailed("MEC204")
        self.assertEqual(res.detected_intent, QueryIntent.COURSE_CODE.value)
        self.assertEqual(res.routing_decision, RoutingAction.DIRECT.value)
        self.assertFalse(res.reformulation_invoked)
        self.reformulator.generate_variants.assert_not_called()
        self.hybrid_retriever.retrieve.assert_called_once()
        self.assertEqual(res.results[0].chunk_id, "direct_chunk")
