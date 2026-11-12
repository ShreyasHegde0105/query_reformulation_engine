"""Unit tests for Phase 2.3 Contextual Cross-Encoder Reranking and Candidate Pool Slicing."""

import unittest
from unittest.mock import MagicMock, patch

from reformulation.cross_encoder import CrossEncoderReranker, build_contextual_passage
from reformulation.search_pipeline import SearchPipeline
from retrieval.models import RetrievedChunk


class TestContextualPassageConstruction(unittest.TestCase):
    """Verify structured passage formatting, missing field handling, and truncation."""

    def _make_chunk(self, chunk_id="c1", title="Hostel Regulations", section="Silence Hours", text="Students must be quiet."):
        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id="doc1",
            title=title,
            domain="Residential",
            chunk_index=0,
            chunk_text=text,
            section=section,
            token_count=10,
            score=0.9,
        )

    def test_complete_metadata_formatting(self):
        chunk = self._make_chunk()
        passage = build_contextual_passage(chunk)
        expected = "Title: Hostel Regulations\nSection: Silence Hours\nContent: Students must be quiet."
        self.assertEqual(passage, expected)

    def test_missing_section_formatting(self):
        chunk = self._make_chunk(section=None)
        passage = build_contextual_passage(chunk)
        expected = "Title: Hostel Regulations\nContent: Students must be quiet."
        self.assertEqual(passage, expected)
        self.assertNotIn("None", passage)

    def test_missing_title_formatting(self):
        chunk = self._make_chunk(title="", section="Exam Guidelines")
        passage = build_contextual_passage(chunk)
        expected = "Section: Exam Guidelines\nContent: Students must be quiet."
        self.assertEqual(passage, expected)
        self.assertNotIn("Title", passage)

    def test_missing_both_title_and_section(self):
        chunk = self._make_chunk(title="", section=None, text="Plain chunk text only.")
        passage = build_contextual_passage(chunk)
        self.assertEqual(passage, "Plain chunk text only.")

    def test_character_truncation(self):
        long_text = "word " * 500  # 2500 characters
        chunk = self._make_chunk(text=long_text)
        passage = build_contextual_passage(chunk, max_characters=100)
        self.assertLessEqual(len(passage), 103)  # 100 + "..."
        self.assertTrue(passage.endswith("..."))


class TestCrossEncoderRerankingLogic(unittest.TestCase):
    """Verify candidate pool slicing, cross-encoder scoring, and error fallbacks."""

    def _make_chunk(self, chunk_id: str, score: float = 0.5, title: str = "T", section: str = "S"):
        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id="doc1",
            title=title,
            domain="D",
            chunk_index=0,
            chunk_text=f"Text for {chunk_id}",
            section=section,
            token_count=10,
            score=score,
        )

    def setUp(self):
        self.reranker = CrossEncoderReranker(enabled=True, contextual=True)
        self.mock_model = MagicMock()
        self.reranker._model = self.mock_model

    def test_candidate_ordering_by_ce_score(self):
        candidates = [
            self._make_chunk("c1", 0.9),
            self._make_chunk("c2", 0.8),
            self._make_chunk("c3", 0.7),
        ]
        # Cross encoder scores c2 highest, then c3, then c1
        self.mock_model.predict.return_value = [0.1, 0.95, 0.4]

        reranked = self.reranker.rerank("query", candidates)
        self.assertEqual([c.chunk_id for c in reranked], ["c2", "c3", "c1"])
        self.assertEqual(reranked[0].metadata["ce_score"], 0.95)
        # Original RRF score is preserved
        self.assertEqual(reranked[0].score, 0.8)

    def test_top_k_candidate_pool_slicing(self):
        candidates = [
            self._make_chunk(f"c{i}", 1.0 / (i + 1)) for i in range(1, 6)
        ]
        # Only rerank top 2
        self.mock_model.predict.return_value = [0.2, 0.8]  # c2 beats c1

        reranked = self.reranker.rerank("query", candidates, top_k=2)
        # c2 moves to rank 1, c1 moves to rank 2; c3, c4, c5 remain unranked in tail
        self.assertEqual([c.chunk_id for c in reranked], ["c2", "c1", "c3", "c4", "c5"])

    def test_stable_fallback_on_scoring_exception(self):
        candidates = [self._make_chunk("c1"), self._make_chunk("c2")]
        self.mock_model.predict.side_effect = RuntimeError("GPU OOM / inference crash")

        # Must not raise; must return original candidates unchanged
        reranked = self.reranker.rerank("query", candidates)
        self.assertEqual([c.chunk_id for c in reranked], ["c1", "c2"])

    def test_disabled_reranker_returns_unmodified(self):
        disabled_reranker = CrossEncoderReranker(enabled=False)
        candidates = [self._make_chunk("c1"), self._make_chunk("c2")]
        self.assertEqual(disabled_reranker.rerank("query", candidates), candidates)


class TestSearchPipelineRerankerIntegration(unittest.TestCase):
    """Verify SearchPipeline properly delegates to CrossEncoderReranker."""

    def test_pipeline_invokes_reranker(self):
        reformulator = MagicMock()
        reformulator.generate_variants.return_value = []
        sem = MagicMock()
        sem.filter_redundant.side_effect = lambda qs, original_query=None: qs
        hybrid = MagicMock()
        chunk1 = RetrievedChunk(chunk_id="c1", document_id="d1", title="T", domain="D", chunk_index=0, chunk_text="t1", section=None, token_count=10, score=0.9)
        chunk2 = RetrievedChunk(chunk_id="c2", document_id="d1", title="T", domain="D", chunk_index=0, chunk_text="t2", section=None, token_count=10, score=0.8)
        hybrid.retrieve.return_value = [chunk1, chunk2]

        reranker = MagicMock(spec=CrossEncoderReranker)
        reranker.enabled = True
        # Reranker reverses order
        reranker.rerank.return_value = [chunk2, chunk1]

        pipeline = SearchPipeline(
            reformulator=reformulator,
            semantic_similarity=sem,
            hybrid_retriever=hybrid,
            reranker=reranker,
            rerank_top_k=10,
            max_workers=1,
        )

        res = pipeline.search_detailed("fee refund")
        self.assertEqual([c.chunk_id for c in res.results], ["c2", "c1"])
        self.assertIn("reranking", res.timings_ms)
        reranker.rerank.assert_called_once()
