import unittest
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from api.app import create_app
from reformulation.answer_generator import Answer, AnswerGenerator
from reformulation.exceptions import QueryValidationError
from reformulation.search_pipeline import SearchPipeline, SearchResult
from retrieval.models import RetrievedChunk


def _chunk(cid, score):
    return RetrievedChunk(chunk_id=cid, document_id=cid.split("_")[0], title="Fee Policy", domain="Finance",
                          chunk_index=0, chunk_text=f"text of {cid}", section="2. FEES", token_count=5, score=score)


class TestSearchAPI(unittest.TestCase):
    def setUp(self):
        self.pipeline = MagicMock(spec=SearchPipeline)
        self.pipeline.search_detailed.return_value = SearchResult(
            results=[_chunk("FIN_CHUNK_001", 0.09), _chunk("FIN_CHUNK_002", 0.05)],
            original_query="semester fees",
            normalized_query="Semester fees",
            reformulations=["semester tuition fee"],
            queries_used=["Semester fees", "semester tuition fee"],
            timings_ms={"total": 12.0},
        )
        self.answerer = MagicMock(spec=AnswerGenerator)
        self.answerer.generate.return_value = Answer(text="Fees are due each semester.", citations=["FIN_CHUNK_001"], answerable=True)
        self.client_cm = TestClient(create_app(pipeline=self.pipeline, components={"keyword": "bm25"},
                                               answer_generator=self.answerer))
        self.client = self.client_cm.__enter__()

    def tearDown(self):
        self.client_cm.__exit__(None, None, None)

    def test_health(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok", "components": {"keyword": "bm25"}})

    def test_search_returns_ranked_hits(self):
        r = self.client.post("/search", json={"query": "  semester fees ", "top_k": 5, "domain": "Finance"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["query"], "semester fees")
        self.assertEqual([h["rank"] for h in body["results"]], [1, 2])
        self.assertEqual(body["results"][0]["chunk_id"], "FIN_CHUNK_001")
        self.assertEqual(body["results"][0]["text"], "text of FIN_CHUNK_001")
        self.assertIsNone(body["trace"])
        self.assertIsNone(body["answer"])
        self.answerer.generate.assert_not_called()
        self.pipeline.search_detailed.assert_called_once_with(
            "semester fees", max_variants=5, filter_domain="Finance", final_k=5
        )

    def test_debug_includes_trace_without_results(self):
        body = self.client.post("/search", json={"query": "semester fees", "debug": True}).json()
        self.assertEqual(body["trace"]["queries_used"], ["Semester fees", "semester tuition fee"])
        self.assertNotIn("results", body["trace"])

    def test_answer_generated_from_results(self):
        body = self.client.post("/search", json={"query": "semester fees", "answer": True}).json()
        self.assertEqual(body["answer"], {"text": "Fees are due each semester.", "citations": ["FIN_CHUNK_001"], "answerable": True})
        question, chunks = self.answerer.generate.call_args.args
        self.assertEqual(question, "semester fees")
        self.assertEqual([c.chunk_id for c in chunks], ["FIN_CHUNK_001", "FIN_CHUNK_002"])

    def test_answer_failure_still_returns_results(self):
        self.answerer.generate.side_effect = RuntimeError("secret provider detail")
        body = self.client.post("/search", json={"query": "semester fees", "answer": True}).json()
        self.assertEqual(len(body["results"]), 2)
        self.assertIsNone(body["answer"])
        self.assertEqual(body["answer_error"], "Answer generation failed (RuntimeError).")

    def test_rate_limit_error_is_short_and_does_not_leak_provider_body(self):
        from reformulation.llm.exceptions import LLMRateLimitError
        self.answerer.generate.side_effect = LLMRateLimitError('Rate limit exceeded (HTTP 429): {"error": {"quotaMetric": "..."}}')
        body = self.client.post("/search", json={"query": "semester fees", "answer": True}).json()
        self.assertIn("rate-limited", body["answer_error"])
        self.assertNotIn("quotaMetric", body["answer_error"])

    def test_answer_requested_without_llm(self):
        with TestClient(create_app(pipeline=self.pipeline)) as client:
            body = client.post("/search", json={"query": "semester fees", "answer": True}).json()
        self.assertIsNone(body["answer"])
        self.assertIn("not configured", body["answer_error"])

    def test_invalid_requests_rejected(self):
        for payload in ({}, {"query": ""}, {"query": "   "}, {"query": "x", "top_k": 0},
                        {"query": "x", "top_k": 51}, {"query": "x" * 501}):
            self.assertEqual(self.client.post("/search", json=payload).status_code, 422, payload)

    def test_pipeline_value_error_is_422(self):
        self.pipeline.search_detailed.side_effect = QueryValidationError("Query contains no searchable content")
        r = self.client.post("/search", json={"query": "???"})
        self.assertEqual(r.status_code, 422)
        self.assertIn("no searchable content", r.json()["detail"])

    def test_query_limit_matches_validator(self):
        from reformulation.validator import MAX_QUERY_LENGTH
        ok = self.client.post("/search", json={"query": "x" * MAX_QUERY_LENGTH})
        self.assertEqual(ok.status_code, 200)
        too_long = self.client.post("/search", json={"query": "x" * (MAX_QUERY_LENGTH + 1)})
        self.assertEqual(too_long.status_code, 422)

    def test_internal_value_error_is_503_without_detail(self):
        self.pipeline.search_detailed.side_effect = ValueError("k must be a positive integer.")
        r = self.client.post("/search", json={"query": "semester fees"})
        self.assertEqual(r.status_code, 503)
        self.assertNotIn("positive integer", r.text)

    def test_degraded_flag_returned(self):
        self.pipeline.search_detailed.return_value.degraded = True
        body = self.client.post("/search", json={"query": "semester fees"}).json()
        self.assertTrue(body["degraded"])
        healthy = SearchResult(results=[], original_query="q", normalized_query="Q", reformulations=[],
                               queries_used=["Q"], timings_ms={})
        self.pipeline.search_detailed.return_value = healthy
        self.assertFalse(self.client.post("/search", json={"query": "q"}).json()["degraded"])

    def test_all_backends_down_is_503_with_real_pipeline(self):
        from reformulation.reformulator import QueryReformulator
        from reformulation.semantic_similarity import SemanticSimilarity
        from retrieval.hybrid_retriever import HybridRetriever
        vec, kw = MagicMock(), MagicMock()
        vec.retrieve.side_effect = RuntimeError("supabase down")
        kw.retrieve.side_effect = RuntimeError("bm25 down")
        reformulator = MagicMock(spec=QueryReformulator)
        reformulator.generate_variants.return_value = []
        sem = MagicMock(spec=SemanticSimilarity)
        sem.filter_redundant.side_effect = lambda qs, original_query=None: qs
        pipeline = SearchPipeline(reformulator, sem, HybridRetriever(vector_retriever=vec, keyword_retriever=kw))
        with TestClient(create_app(pipeline=pipeline)) as client:
            r = client.post("/search", json={"query": "semester fees"})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["detail"], "Search backend unavailable")

    def test_real_pipeline_raises_query_validation_error(self):
        # The API maps exactly this exception type to 422.
        from reformulation.validator import MAX_QUERY_LENGTH
        pipeline = SearchPipeline(MagicMock(), MagicMock(), MagicMock())
        with self.assertRaises(QueryValidationError):
            pipeline.search_detailed("x" * (MAX_QUERY_LENGTH + 1))

    def test_backend_failure_is_503(self):
        self.pipeline.search_detailed.side_effect = RuntimeError("supabase down")
        r = self.client.post("/search", json={"query": "semester fees"})
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json()["detail"], "Search backend unavailable")


if __name__ == "__main__":
    unittest.main()
