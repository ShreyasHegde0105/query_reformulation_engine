from __future__ import annotations

import json
import math
import os
import pytest
from typing import List

from evaluation.metrics import (
    calculate_all_metrics,
    calculate_dcg_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
)
from evaluation.schema import (
    EvaluationConfig,
    EvaluationSummary,
    GoldenQuery,
    QueryResult,
)
from evaluation.evaluator import RetrievalEvaluator
from evaluation.run_evaluation import (
    generate_markdown_report,
    load_golden_dataset,
    save_json_results,
)
from retrieval.models import RetrievedChunk


# ============================================================================
# 1. Metric Unit Tests
# ============================================================================

class TestEvaluationMetrics:
    """Test suite for standard IR metric calculations and edge cases."""

    def test_recall_at_k_standard(self):
        relevant = {"c1", "c2"}
        # Both found in top 5
        assert calculate_recall_at_k(["c1", "c2", "c3", "c4", "c5"], relevant, k=5) == 1.0
        # 1 of 2 found in top 5
        assert calculate_recall_at_k(["c1", "c3", "c4", "c5", "c6"], relevant, k=5) == 0.5
        # 0 of 2 found
        assert calculate_recall_at_k(["c3", "c4", "c5"], relevant, k=5) == 0.0

    def test_recall_at_k_rank_cutoff(self):
        relevant = {"c1", "c2"}
        # c2 is at rank 6, so at k=5 only c1 is found
        retrieved = ["c1", "c3", "c4", "c5", "c6", "c2"]
        assert calculate_recall_at_k(retrieved, relevant, k=5) == 0.5
        assert calculate_recall_at_k(retrieved, relevant, k=6) == 1.0
        assert calculate_recall_at_k(retrieved, relevant, k=10) == 1.0

    def test_recall_at_k_edge_cases(self):
        # Empty relevant set -> 0.0 (no division by zero)
        assert calculate_recall_at_k(["c1", "c2"], set(), k=5) == 0.0
        # Empty retrieved list -> 0.0
        assert calculate_recall_at_k([], {"c1"}, k=5) == 0.0
        # k <= 0 -> 0.0
        assert calculate_recall_at_k(["c1", "c2"], {"c1"}, k=0) == 0.0
        assert calculate_recall_at_k(["c1", "c2"], {"c1"}, k=-1) == 0.0
        # Duplicate items in retrieved list are deduplicated preserving order
        assert calculate_recall_at_k(["c1", "c1", "c2", "c2"], {"c1", "c2"}, k=2) == 1.0

    def test_mrr_calculation(self):
        relevant = {"c1", "c2"}
        # First relevant at rank 1 -> 1.0
        assert calculate_mrr(["c1", "c3", "c4"], relevant) == 1.0
        # First relevant at rank 2 -> 0.5
        assert calculate_mrr(["c3", "c1", "c4"], relevant) == 0.5
        # First relevant at rank 4 -> 0.25
        assert calculate_mrr(["c3", "c4", "c5", "c2"], relevant) == 0.25
        # None found -> 0.0
        assert calculate_mrr(["c3", "c4", "c5"], relevant) == 0.0
        # Empty inputs -> 0.0
        assert calculate_mrr([], relevant) == 0.0
        assert calculate_mrr(["c1"], set()) == 0.0

    def test_dcg_and_ndcg_binary(self):
        relevance = {"c1": 1, "c2": 1}
        # Perfect retrieval: c1, c2 in top 2 -> NDCG = 1.0
        assert calculate_ndcg_at_k(["c1", "c2", "c3"], relevance, k=5) == pytest.approx(1.0)
        # Irrelevant results -> 0.0
        assert calculate_ndcg_at_k(["c3", "c4", "c5"], relevance, k=5) == 0.0
        # Sub-optimal ranking: c1 at rank 3
        ndcg = calculate_ndcg_at_k(["c3", "c4", "c1"], relevance, k=5)
        assert 0.0 < ndcg < 1.0

    def test_dcg_and_ndcg_graded(self):
        # Graded relevance: c1 is highly relevant (3), c2 is relevant (1)
        relevance = {"c1": 3, "c2": 1}
        # Ideal ordering: c1 (rank 1), c2 (rank 2)
        assert calculate_ndcg_at_k(["c1", "c2"], relevance, k=2) == pytest.approx(1.0)
        # Sub-ideal ordering: c2 (rank 1), c1 (rank 2)
        ndcg_sub = calculate_ndcg_at_k(["c2", "c1"], relevance, k=2)
        assert 0.0 < ndcg_sub < 1.0
        # DCG of c2 then c1 is lower than c1 then c2
        assert calculate_dcg_at_k(["c2", "c1"], relevance, k=2) < calculate_dcg_at_k(["c1", "c2"], relevance, k=2)

    def test_calculate_all_metrics_bundle(self):
        retrieved = ["c1", "c3", "c2", "c4", "c5"]
        relevant = ["c1", "c2"]
        graded = {"c1": 3, "c2": 2}
        
        metrics = calculate_all_metrics(retrieved, relevant, graded_relevance=graded, k_list=[5, 10])
        assert "recall@5" in metrics
        assert "recall@10" in metrics
        assert "mrr" in metrics
        assert "ndcg@5" in metrics
        assert "ndcg@10" in metrics
        
        assert metrics["recall@5"] == 1.0
        assert metrics["mrr"] == 1.0
        assert 0.0 < metrics["ndcg@5"] <= 1.0


# ============================================================================
# 2. Golden Dataset Schema & Loading Tests
# ============================================================================

class TestGoldenDatasetSchema:
    """Test suite for golden query schema parsing, validation, and dataset integrity."""

    def test_golden_query_creation_and_validation(self):
        q = GoldenQuery(
            query_id="Q001",
            query="semester fees",
            category="keyword",
            relevant_chunks=["FIN-POL-001_CHUNK_001"],
            graded_relevance={"FIN-POL-001_CHUNK_001": 3},
            tags=["finance", "fees"],
            target_doc_id="FIN-POL-001",
            description="Semester fee breakdown",
        )
        assert q.query_id == "Q001"
        assert q.relevant_chunk_set == {"FIN-POL-001_CHUNK_001"}
        
        # Test serialization round-trip
        data = q.to_dict()
        q_loaded = GoldenQuery.from_dict(data)
        assert q_loaded.query_id == q.query_id
        assert q_loaded.query == q.query
        assert q_loaded.relevant_chunks == q.relevant_chunks

    def test_golden_query_invalid_params(self):
        with pytest.raises(ValueError):
            GoldenQuery(query_id="", query="valid", category="keyword", relevant_chunks=["c1"])
        with pytest.raises(ValueError):
            GoldenQuery(query_id="Q1", query="", category="keyword", relevant_chunks=["c1"])
        with pytest.raises(ValueError):
            GoldenQuery(query_id="Q1", query="valid", category="", relevant_chunks=["c1"])

    def test_load_production_golden_dataset(self):
        dataset = load_golden_dataset()
        assert len(dataset) == 50, f"Expected 50 golden queries, found {len(dataset)}"
        
        # Check all 10 categories have exactly 5 queries each
        categories = {}
        total_chunks = 0
        for q in dataset:
            categories[q.category] = categories.get(q.category, 0) + 1
            total_chunks += len(q.relevant_chunks)
            assert len(q.relevant_chunks) >= 1, f"Query {q.query_id} has no relevant chunks"

        expected_categories = [
            "keyword", "natural_language", "abbreviation", "typo", "course_code",
            "numeric", "policy", "eligibility", "procedure", "lexical_diversity"
        ]
        for ec in expected_categories:
            assert categories.get(ec) == 5, f"Category '{ec}' expected 5 queries, got {categories.get(ec)}"

        assert total_chunks == 100, f"Expected 100 relevant chunks total, got {total_chunks}"

    def test_golden_chunks_exist_in_corpus(self):
        dataset = load_golden_dataset()
        chunks_path = os.path.join(os.path.dirname(__file__), "..", "Data", "chunks", "chunks.json")
        assert os.path.exists(chunks_path), "Data/chunks/chunks.json must exist"
        
        with open(chunks_path, "r", encoding="utf-8") as f:
            corpus_chunks = json.load(f)
        corpus_ids = {c["chunk_id"] for c in corpus_chunks}

        missing = []
        for q in dataset:
            for cid in q.relevant_chunks:
                if cid not in corpus_ids:
                    missing.append((q.query_id, cid))

        assert len(missing) == 0, f"Ground truth contains chunks not in corpus: {missing}"


# ============================================================================
# 3. Evaluator Mock & Pipeline Tests
# ============================================================================

class MockRetriever:
    """Mock retriever returning deterministic chunk results."""
    def __init__(self, return_map: dict):
        self.return_map = return_map

    def retrieve(self, query: str, **kwargs) -> List[RetrievedChunk]:
        chunk_ids = self.return_map.get(query, ["FALLBACK_CHUNK"])
        return [
            RetrievedChunk(
                chunk_id=cid,
                document_id="DOC-1",
                title="Mock Doc",
                domain="Mock Domain",
                chunk_index=0,
                chunk_text="Mock Text",
                section=None,
                token_count=10,
                score=0.9,
            )
            for cid in chunk_ids
        ]


class TestRetrievalEvaluator:
    """Test suite for the multi-variant RetrievalEvaluator."""

    def test_custom_variant_registration_and_evaluation(self):
        dataset = [
            GoldenQuery(
                query_id="Q001",
                query="semester fees",
                category="keyword",
                relevant_chunks=["C1", "C2"],
            ),
            GoldenQuery(
                query_id="Q002",
                query="hostel allotment",
                category="keyword",
                relevant_chunks=["C3"],
            ),
        ]

        evaluator = RetrievalEvaluator()
        
        # Register custom mock variant
        evaluator.register_variant("mock_perfect", lambda q, cfg: q.relevant_chunks)
        evaluator.register_variant("mock_half", lambda q, cfg: [q.relevant_chunks[0], "RANDOM_CHUNK"])

        summaries = evaluator.evaluate_all_variants(dataset, ["mock_perfect", "mock_half"])
        
        assert "mock_perfect" in summaries
        assert summaries["mock_perfect"].mean_metrics["recall@5"] == 1.0
        assert summaries["mock_perfect"].mean_metrics["mrr"] == 1.0

        assert "mock_half" in summaries
        # Q001 gets 1/2 recall = 0.5, Q002 gets 1/1 recall = 1.0 -> mean = 0.75
        assert summaries["mock_half"].mean_metrics["recall@5"] == 0.75
        assert summaries["mock_half"].mean_metrics["mrr"] == 1.0

    def test_report_generation(self, tmp_path):
        summary = EvaluationSummary(
            variant="baseline_vector",
            total_queries=2,
            mean_metrics={"recall@5": 0.8, "recall@10": 0.9, "mrr": 0.85, "ndcg@5": 0.82, "ndcg@10": 0.88},
            category_metrics={
                "keyword": {"recall@5": 0.8, "recall@10": 0.9, "mrr": 0.85, "ndcg@5": 0.82}
            },
            per_query_results=[],
        )
        summaries = {"baseline_vector": summary}

        md_path = str(tmp_path / "report.md")
        json_path = str(tmp_path / "results.json")

        generate_markdown_report(summaries, md_path)
        save_json_results(summaries, json_path)

        assert os.path.exists(md_path)
        assert os.path.exists(json_path)

        with open(md_path, "r", encoding="utf-8") as f:
            md_text = f.read()
        assert "| Original + Vector | 0.8000 | 0.9000 | 0.8500 | 0.8200 | 0.8800 |" in md_text
