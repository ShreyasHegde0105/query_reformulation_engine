"""The demo must not judge relevance without labels and must not declare a winner."""

import contextlib
import io
import re
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from reformulation.production_reranker import ProductionReranker, RerankerConfig
from reformulation.search_pipeline import SearchPipeline
from reformulation.semantic_similarity import FilterDecision
from retrieval.models import RetrievedChunk

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = re.compile(r"judged relevant|improved judged|reduced judged|reformulation improved|"
                       r"reformulation reduced|score > 0|counts as relevant|probabilit(y|ies) of relevance", re.I)


def chunk(i, doc=None):
    return RetrievedChunk(chunk_id=f"C{i}", document_id=doc or f"D{i % 7}", title=f"T{i}", domain="X",
                          chunk_index=i, chunk_text=f"text {i}", section=None, token_count=5, score=1.0 / (i + 1))


class FakeModel:
    def predict(self, pairs):
        return [-float(p[1].split("text ")[1]) for p in pairs]  # logits are all <= 0, like the reported case


def make_demo(golden=None, rerank=False):
    import jury_demo
    demo = object.__new__(jury_demo.PresentationDemo)
    demo.use_cache, demo.enable_judge = False, False
    demo.reformulator = MagicMock()
    demo.reformulator.generate_variants.return_value = ["variant one", "variant two"]
    demo.semantic_similarity = MagicMock()
    demo.semantic_similarity.filter_redundant_with_diagnostics.side_effect = lambda qs, original_query=None: (
        qs, [FilterDecision(query=q, similarity=1.0, kept=True, reason="kept") for q in qs])
    hybrid = MagicMock()
    hybrid.retrieve.side_effect = lambda **kw: [chunk(i + (3 if kw["query"] != "Fee refund" else 0))
                                                for i in range(kw["final_k"])]
    demo.hybrid_retriever = hybrid
    demo.spell_corrector = None
    demo.pipeline = SearchPipeline(reformulator=demo.reformulator, semantic_similarity=MagicMock(),
                                   hybrid_retriever=hybrid, max_workers=1, intent_router=MagicMock())
    demo.reranker = ProductionReranker(RerankerConfig(enabled=rerank, candidate_limit=50), model=FakeModel())
    demo.golden_queries = golden or []
    demo._golden_by_id = {q["query_id"]: q for q in demo.golden_queries}
    demo._golden_by_text = {q["query"].strip().lower(): q for q in demo.golden_queries}
    demo.reformulation_cache, demo._live_reformulations = {}, {}
    return demo


def run(demo, query):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = demo.execute_demo_query(query)
    return buf.getvalue(), results


class TestDemoEvaluation(unittest.TestCase):
    def test_source_has_no_verdict_or_threshold(self):
        src = (ROOT / "jury_demo.py").read_text(encoding="utf-8")
        for bad in ("JUDGE_RELEVANT_THRESHOLD", "_judge_relevance", "Judged relevant", "Verdict"):
            self.assertNotIn(bad, src)

    def test_unlabelled_query_is_not_judged(self):
        out, results = run(make_demo(rerank=True), "fee refund")
        self.assertIn("Relevance: NOT EVALUATED", out)
        self.assertIn("Unanswerable-query detection: NOT AVAILABLE", out)
        self.assertIn("not probabilities", out)
        self.assertIsNone(FORBIDDEN.search(out), FORBIDDEN.search(out))
        self.assertEqual(len(results), 20)

    def test_labelled_query_reports_metrics_and_latency(self):
        golden = [{"query_id": "Q1", "query": "fee refund", "category": "x", "relevant_chunks": ["C0", "C4"],
                   "graded_relevance": {"C0": 3, "C4": 2}, "description": "refund tiers"}]
        out, _ = run(make_demo(golden=golden, rerank=True), "fee refund")
        for needed in ("Precision@5", "Precision@10", "Recall@10", "MRR@10", "NDCG@10", "Reranked",
                       "Label note (evidence for the grades): refund tiers", "label provenance is not recorded",
                       "Latency: original retrieval", "reranking", "One query cannot show"):
            self.assertIn(needed, out)
        self.assertIsNone(FORBIDDEN.search(out))

    def test_reranker_disabled_status(self):
        out, _ = run(make_demo(rerank=False), "fee refund")
        self.assertIn("Status: DISABLED", out)
        self.assertNotIn("Phase 4: pure Cross-Encoder", out)   # stale claims removed


if __name__ == "__main__":
    unittest.main()
