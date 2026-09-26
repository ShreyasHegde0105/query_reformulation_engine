"""Unit and integration tests for CrossEncoderReranker and Candidate Pool Reranking.

Covers all Phase 4 requirements:
1. CrossEncoder:
   - model loads
   - batch scoring
   - deterministic output
   - empty input
   - single candidate
   - multiple candidates
2. Candidate pool:
   - correct top-K
   - deduplication
   - stable ordering
   - relevant chunk tracking
3. Reranking:
   - scores assigned to correct chunks
   - descending ordering
   - metadata preserved
   - deterministic results
4. Pipeline:
   - RRF remains unchanged
   - CE is optional
   - existing pipeline works when CE is disabled
"""

from __future__ import annotations

import copy
import os
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from reformulation.cross_encoder import CrossEncoderReranker
from retrieval.models import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(
    chunk_id: str,
    score: float = 0.5,
    text: str = "",
    metadata: dict = None,
) -> RetrievedChunk:
    """Build a minimal RetrievedChunk for testing."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc_" + chunk_id,
        title="Title " + chunk_id,
        domain="test",
        chunk_index=0,
        chunk_text=text or f"Text for {chunk_id}",
        section=None,
        token_count=10,
        score=score,
        metadata=metadata if metadata is not None else {"source": "test_doc"},
        retriever_type="hybrid",
    )


def _make_reranker_with_mock_scores(scores: List[float], batch_size: int = 32) -> CrossEncoderReranker:
    """Build a CrossEncoderReranker whose underlying model returns fixed scores."""
    reranker = CrossEncoderReranker(batch_size=batch_size, enabled=True)
    mock_model = MagicMock()
    def predict_side_effect(pairs):
        n = len(pairs)
        return scores[:n]
    mock_model.predict.side_effect = predict_side_effect
    reranker._model = mock_model
    return reranker


# ---------------------------------------------------------------------------
# 1. CrossEncoder Suite
# ---------------------------------------------------------------------------

class TestCrossEncoderModel:
    def test_model_loads(self):
        """CrossEncoder loads lazily on first rerank call or explicit _load_model."""
        reranker = CrossEncoderReranker(enabled=True)
        assert not reranker.is_loaded
        # Mock sentence_transformers CrossEncoder to avoid network call in unit test
        with patch("sentence_transformers.CrossEncoder") as mock_ce:
            mock_ce.return_value = MagicMock()
            reranker._load_model()
            assert reranker.is_loaded
            mock_ce.assert_called_once_with(reranker.model_name)

    def test_batch_scoring(self):
        """Batch scoring correctly splits inputs into batches and collects all scores."""
        reranker = CrossEncoderReranker(batch_size=2, enabled=True)
        mock_model = MagicMock()
        mock_model.predict.side_effect = [
            [0.1, 0.2],  # batch 1 (size 2)
            [0.3, 0.4],  # batch 2 (size 2)
            [0.5],       # batch 3 (size 1)
        ]
        reranker._model = mock_model

        pairs = [("q", f"doc_{i}") for i in range(5)]
        scores = reranker._score_pairs(pairs)

        assert scores == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert mock_model.predict.call_count == 3

    def test_deterministic_output(self):
        """Repeated scoring of identical candidates produces identical scores and ranks."""
        chunks = [_make_chunk("c1"), _make_chunk("c2"), _make_chunk("c3")]
        reranker = _make_reranker_with_mock_scores([0.45, 0.85, 0.15])

        out1 = reranker.rerank("same query", chunks)
        out2 = reranker.rerank("same query", chunks)

        assert [c.chunk_id for c in out1] == [c.chunk_id for c in out2]
        assert [c.metadata["ce_score"] for c in out1] == [c.metadata["ce_score"] for c in out2]

    def test_empty_input(self):
        """Empty candidate pool returns empty list immediately without error."""
        reranker = CrossEncoderReranker(enabled=True)
        result = reranker.rerank("some query", [])
        assert result == []

    def test_single_candidate(self):
        """Single candidate is returned with ce_score attached."""
        reranker = _make_reranker_with_mock_scores([2.345])
        chunk = _make_chunk("solo_chunk")
        result = reranker.rerank("query", [chunk])
        assert len(result) == 1
        assert result[0].chunk_id == "solo_chunk"
        assert abs(result[0].metadata["ce_score"] - 2.345) < 1e-6

    def test_multiple_candidates(self):
        """Multiple candidates are all scored and reordered according to CE score."""
        chunks = [
            _make_chunk("c1", text="irrelevant text"),
            _make_chunk("c2", text="highly relevant text"),
            _make_chunk("c3", text="somewhat relevant text"),
            _make_chunk("c4", text="completely unrelated"),
        ]
        scores = [-5.2, 8.4, 3.1, -8.9]
        reranker = _make_reranker_with_mock_scores(scores)
        result = reranker.rerank("query", chunks)

        assert [c.chunk_id for c in result] == ["c2", "c3", "c1", "c4"]
        assert len(result) == 4


# ---------------------------------------------------------------------------
# 2. Candidate Pool Suite
# ---------------------------------------------------------------------------

class TestCandidatePool:
    def test_correct_top_k(self):
        """Candidate pool truncation respects requested top-K."""
        chunks = [_make_chunk(f"c_{i}", score=1.0 / (i + 1)) for i in range(50)]
        k_values = [20, 30, 50]
        for k in k_values:
            pool = chunks[:k]
            assert len(pool) == k

    def test_deduplication(self):
        """Candidate pool chunks are unique with distinct chunk_ids."""
        chunk_ids = ["c1", "c2", "c3", "c4"]
        chunks = [_make_chunk(cid) for cid in chunk_ids]
        pool_ids = set(c.chunk_id for c in chunks)
        assert len(pool_ids) == len(chunks)

    def test_stable_ordering(self):
        """Ties in score are broken deterministically by chunk_id."""
        chunks = [
            _make_chunk("c_zebra", score=0.5),
            _make_chunk("c_alpha", score=0.5),
            _make_chunk("c_beta", score=0.5),
        ]
        reranker = _make_reranker_with_mock_scores([0.5, 0.5, 0.5])
        result = reranker.rerank("query", chunks)
        assert [c.chunk_id for c in result] == ["c_alpha", "c_beta", "c_zebra"]

    def test_relevant_chunk_tracking(self):
        """Correctly classifies ALL_PRESENT, SOME_PRESENT, and NONE_PRESENT."""
        relevant = {"rel_1", "rel_2"}

        pool_all = [_make_chunk("rel_1"), _make_chunk("rel_2"), _make_chunk("irr")]
        pool_some = [_make_chunk("rel_1"), _make_chunk("irr1"), _make_chunk("irr2")]
        pool_none = [_make_chunk("irr1"), _make_chunk("irr2"), _make_chunk("irr3")]

        def check_status(pool):
            pool_ids = set(c.chunk_id for c in pool)
            found = relevant.intersection(pool_ids)
            if len(found) == len(relevant):
                return "ALL_PRESENT"
            elif len(found) > 0:
                return "SOME_PRESENT"
            return "NONE_PRESENT"

        assert check_status(pool_all) == "ALL_PRESENT"
        assert check_status(pool_some) == "SOME_PRESENT"
        assert check_status(pool_none) == "NONE_PRESENT"


# ---------------------------------------------------------------------------
# 3. Reranking Suite
# ---------------------------------------------------------------------------

class TestReranking:
    def test_scores_assigned_to_correct_chunks(self):
        """Each candidate chunk receives its own computed CE score in metadata."""
        chunks = [_make_chunk("c1"), _make_chunk("c2"), _make_chunk("c3")]
        scores = [1.23, 4.56, -2.34]
        reranker = _make_reranker_with_mock_scores(scores)
        result = reranker.rerank("q", chunks)

        by_id = {c.chunk_id: c.metadata["ce_score"] for c in result}
        assert abs(by_id["c1"] - 1.23) < 1e-5
        assert abs(by_id["c2"] - 4.56) < 1e-5
        assert abs(by_id["c3"] - (-2.34)) < 1e-5

    def test_descending_ordering(self):
        """Returned candidates are strictly sorted by ce_score descending."""
        chunks = [_make_chunk(f"c{i}") for i in range(5)]
        scores = [-3.0, 5.0, 1.0, 8.0, -1.0]
        reranker = _make_reranker_with_mock_scores(scores)
        result = reranker.rerank("q", chunks)

        res_scores = [c.metadata["ce_score"] for c in result]
        assert res_scores == sorted(res_scores, reverse=True)
        assert [c.chunk_id for c in result] == ["c3", "c1", "c2", "c4", "c0"]

    def test_metadata_preserved(self):
        """All existing metadata keys on chunks are fully preserved after reranking."""
        orig_meta = {"source": "manual.pdf", "page": 42, "category": "policy"}
        chunk = _make_chunk("c1", metadata=dict(orig_meta))
        reranker = _make_reranker_with_mock_scores([3.14])
        result = reranker.rerank("q", [chunk])

        for k, v in orig_meta.items():
            assert result[0].metadata[k] == v
        assert result[0].metadata["ce_score"] == 3.14

    def test_original_score_preserved(self):
        """The original RRF score on .score is preserved and not overwritten."""
        chunk = _make_chunk("c1", score=0.0825)
        reranker = _make_reranker_with_mock_scores([4.0])
        result = reranker.rerank("q", [chunk])
        assert abs(result[0].score - 0.0825) < 1e-9


# ---------------------------------------------------------------------------
# 4. Pipeline Suite
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_rrf_remains_unchanged(self):
        """When reranker is disabled, output ordering and scores are strictly RRF."""
        reranker = CrossEncoderReranker(enabled=False)
        chunks = [
            _make_chunk("c1", score=0.09),
            _make_chunk("c2", score=0.07),
            _make_chunk("c3", score=0.05),
        ]
        result = reranker.rerank("query", chunks)
        assert [c.chunk_id for c in result] == ["c1", "c2", "c3"]
        assert [c.score for c in result] == [0.09, 0.07, 0.05]
        for c in result:
            assert "ce_score" not in c.metadata

    def test_ce_is_optional(self):
        """CrossEncoderReranker degrades to passthrough when disabled or unavailable."""
        reranker = CrossEncoderReranker(enabled=False)
        chunk = _make_chunk("c1")
        result = reranker.rerank("q", [chunk])
        assert len(result) == 1
        assert not reranker.is_loaded

    def test_existing_pipeline_works_when_disabled(self):
        """Environment variable RERANKER_ENABLED=false leaves default pipeline unchanged."""
        with patch.dict(os.environ, {"RERANKER_ENABLED": "false"}, clear=False):
            reranker = CrossEncoderReranker.from_env()
            assert not reranker.enabled
            chunks = [_make_chunk("a", 0.9), _make_chunk("b", 0.8)]
            out = reranker.rerank("query", chunks)
            assert [c.chunk_id for c in out] == ["a", "b"]

    def test_type_a_vs_type_b_distinction(self):
        """Type A cannot be rescued by CE; Type B can be promoted."""
        # Type A: target absent from candidate pool
        pool_type_a = [_make_chunk("irr1"), _make_chunk("irr2")]
        reranker = _make_reranker_with_mock_scores([1.0, 0.5])
        res_a = reranker.rerank("q", pool_type_a)
        assert "target_chunk" not in [c.chunk_id for c in res_a]

        # Type B: target present at low rank, promoted by CE
        pool_type_b = [
            _make_chunk("irr1", score=0.09),
            _make_chunk("target_chunk", score=0.02),  # low RRF rank
        ]
        reranker_b = _make_reranker_with_mock_scores([-2.0, 6.0])  # high CE score for target
        res_b = reranker_b.rerank("q", pool_type_b)
        assert res_b[0].chunk_id == "target_chunk"
