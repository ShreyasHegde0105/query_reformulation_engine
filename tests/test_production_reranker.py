"""Tests for the production reranker and its SearchPipeline integration (fake model, no download)."""

import dataclasses
import os
import unittest
from unittest.mock import MagicMock, patch

from reformulation.production_reranker import LOGIT_KEY, ProductionReranker, RerankerConfig
from reformulation.search_pipeline import SearchPipeline
from retrieval.models import RetrievedChunk


def chunk(i, score=1.0, doc=None):
    return RetrievedChunk(chunk_id=f"C{i}", document_id=doc or f"D{i}", title=f"Title {i}", domain="X",
                          chunk_index=i, chunk_text=f"text {i}", section=f"{i}. SECTION", token_count=5,
                          score=score, metadata={"keep": i})


class FakeModel:
    """predict() returns a logit per pair from a lookup on the passage text."""

    def __init__(self, logits_by_index):
        self.logits = logits_by_index
        self.pairs = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        out = []
        for _, passage in pairs:
            idx = int(passage.split("text ")[1].split()[0])
            out.append(self.logits[idx])
        return out


class TestRerankerConfig(unittest.TestCase):
    def test_defaults_off_and_no_threshold(self):
        cfg = RerankerConfig()
        self.assertFalse(cfg.enabled)
        names = {f.name for f in dataclasses.fields(RerankerConfig)}
        self.assertFalse(any("threshold" in n or "prob" in n or "contextual" in n or "text_only" in n for n in names))

    def test_candidate_limits(self):
        for n in (20, 50, 100):
            self.assertEqual(RerankerConfig(candidate_limit=n).candidate_limit, n)
        for bad in (0, 101, 2.5):
            with self.assertRaises(ValueError):
                RerankerConfig(candidate_limit=bad)

    def test_from_env(self):
        env = {"RERANKER_ENABLED": "true", "RERANKER_CANDIDATE_LIMIT": "20", "RERANKER_MODEL": "m/x"}
        with patch.dict(os.environ, env):
            cfg = RerankerConfig.from_env()
        self.assertEqual((cfg.enabled, cfg.candidate_limit, cfg.model_name), (True, 20, "m/x"))
        with patch.dict(os.environ, {"RERANKER_ENABLED": ""}):
            self.assertFalse(RerankerConfig.from_env().enabled)


class TestProductionReranker(unittest.TestCase):
    def test_sorts_by_logit_preserves_ids_and_fusion_score(self):
        model = FakeModel({0: -11.0, 1: 4.2, 2: -3.0})
        rr = ProductionReranker(RerankerConfig(enabled=True, candidate_limit=20), model=model)
        cands = [chunk(0, 0.9), chunk(1, 0.5), chunk(2, 0.3)]
        out = rr.rerank("q", cands)
        self.assertTrue(out.reranked)
        self.assertEqual([c.chunk_id for c in out.results], ["C1", "C2", "C0"])
        self.assertEqual([c.document_id for c in out.results], ["D1", "D2", "D0"])
        # fusion score untouched; raw logit stored unchanged (no sigmoid, no clipping)
        self.assertEqual([c.score for c in out.results], [0.5, 0.3, 0.9])
        self.assertEqual([c.metadata[LOGIT_KEY] for c in out.results], [4.2, -3.0, -11.0])
        self.assertEqual(out.results[0].metadata["keep"], 1)
        self.assertGreaterEqual(out.latency_ms, 0.0)
        self.assertEqual(out.candidates_scored, 3)

    def test_inputs_not_mutated(self):
        cands = [chunk(0), chunk(1)]
        ProductionReranker(RerankerConfig(enabled=True), model=FakeModel({0: 1.0, 1: 2.0})).rerank("q", cands)
        self.assertNotIn(LOGIT_KEY, cands[0].metadata)

    def test_contextual_passages_used(self):
        model = FakeModel({0: 0.0})
        ProductionReranker(RerankerConfig(enabled=True), model=model).rerank("q", [chunk(0)])
        passage = model.pairs[0][1]
        self.assertIn("Title: Title 0", passage)
        self.assertIn("Section: 0. SECTION", passage)
        self.assertIn("Content: text 0", passage)

    def test_candidate_limit_and_final_k(self):
        model = FakeModel({i: float(i) for i in range(30)})
        rr = ProductionReranker(RerankerConfig(enabled=True, candidate_limit=20), model=model)
        out = rr.rerank("q", [chunk(i) for i in range(30)], final_k=5)
        self.assertEqual(len(model.pairs), 20)          # only the first 20 are scored
        self.assertEqual([c.chunk_id for c in out.results], ["C19", "C18", "C17", "C16", "C15"])

    def test_empty_and_errors_are_safe(self):
        rr = ProductionReranker(RerankerConfig(enabled=True), model=FakeModel({}))
        self.assertEqual(rr.rerank("q", []).results, [])

        broken = MagicMock()
        broken.predict.side_effect = RuntimeError("CUDA OOM")
        out = ProductionReranker(RerankerConfig(enabled=True), model=broken).rerank("q", [chunk(0), chunk(1)], final_k=1)
        self.assertFalse(out.reranked)
        self.assertIn("CUDA OOM", out.error)
        self.assertEqual([c.chunk_id for c in out.results], ["C0"])

        wrong_len = MagicMock()
        wrong_len.predict.return_value = [1.0]
        out = ProductionReranker(RerankerConfig(enabled=True), model=wrong_len).rerank("q", [chunk(0), chunk(1)])
        self.assertFalse(out.reranked)

    def test_ties_keep_retrieval_order(self):
        rr = ProductionReranker(RerankerConfig(enabled=True), model=FakeModel({0: 1.0, 1: 1.0, 2: 1.0}))
        out = rr.rerank("q", [chunk(2), chunk(0), chunk(1)])
        self.assertEqual([c.chunk_id for c in out.results], ["C2", "C0", "C1"])


class TestPipelineIntegration(unittest.TestCase):
    def make(self, reranker, n_results=60):
        reformulator = MagicMock()
        reformulator.generate_variants.return_value = []
        sem = MagicMock()
        sem.filter_redundant.side_effect = lambda qs, original_query=None: qs
        hybrid = MagicMock()
        hybrid.retrieve.side_effect = lambda **kw: [chunk(i, 1.0 / (i + 1)) for i in range(min(kw["final_k"], n_results))]
        router = MagicMock()
        from reformulation.intent_router import RoutingAction
        router.route.return_value = MagicMock(action=RoutingAction.REFORMULATE, intent=MagicMock(value="X"), reason="")
        return SearchPipeline(reformulator=reformulator, semantic_similarity=sem, hybrid_retriever=hybrid,
                              reranker=reranker, max_workers=1, intent_router=router), hybrid

    def test_candidate_pool_deeper_than_output(self):
        model = FakeModel({i: float(i) for i in range(60)})
        rr = ProductionReranker(RerankerConfig(enabled=True, candidate_limit=50), model=model)
        p, hybrid = self.make(rr)
        res = p.search_detailed("fee refund", final_k=10, candidate_k=50)
        self.assertEqual(len(res.results), 10)
        self.assertEqual(len(model.pairs), 50)
        self.assertEqual(res.results[0].chunk_id, "C49")      # best logit in the 50-candidate pool
        self.assertIn("reranking", res.timings_ms)
        self.assertTrue(res.reranking["reranked"])
        self.assertIn("logit", res.reranking["score_type"])
        self.assertNotIn("prob", res.reranking["score_type"].replace("not", ""))

    def test_disabled_reranker_changes_nothing(self):
        rr = ProductionReranker(RerankerConfig(enabled=False), model=FakeModel({}))
        p_off, _ = self.make(rr)
        p_none, _ = self.make(None)
        a = p_off.search_detailed("fee refund", final_k=10)
        b = p_none.search_detailed("fee refund", final_k=10)
        self.assertEqual([c.chunk_id for c in a.results], [c.chunk_id for c in b.results])
        self.assertIsNone(a.reranking)
        self.assertNotIn("reranking", a.timings_ms)

    def test_reranker_failure_degrades_safely(self):
        broken = MagicMock()
        broken.predict.side_effect = RuntimeError("boom")
        p, _ = self.make(ProductionReranker(RerankerConfig(enabled=True), model=broken))
        res = p.search_detailed("fee refund", final_k=10)
        self.assertEqual(len(res.results), 10)
        self.assertTrue(res.degraded)
        self.assertFalse(res.reranking["reranked"])


class TestApiReranker(unittest.TestCase):
    def test_api_exposes_logit_only_when_reranked(self):
        from fastapi.testclient import TestClient
        from api.app import create_app

        model = FakeModel({i: float(-i) for i in range(60)})
        p, _ = TestPipelineIntegration().make(ProductionReranker(RerankerConfig(enabled=True, candidate_limit=20), model=model))
        with TestClient(create_app(pipeline=p)) as client:
            body = client.post("/search", json={"query": "fee refund", "top_k": 3, "debug": True}).json()
        self.assertEqual([h["chunk_id"] for h in body["results"]], ["C0", "C1", "C2"])
        self.assertEqual(body["results"][1]["reranker_logit"], -1.0)
        self.assertIn("reranking", body["trace"])

        p2, _ = TestPipelineIntegration().make(None)
        with TestClient(create_app(pipeline=p2)) as client:
            body = client.post("/search", json={"query": "fee refund", "top_k": 3}).json()
        self.assertIsNone(body["results"][0]["reranker_logit"])


if __name__ == "__main__":
    unittest.main()
