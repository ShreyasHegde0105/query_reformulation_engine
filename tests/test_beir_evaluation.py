import json
import os
import tempfile
import unittest

from evaluation.run_beir_evaluation import cache_path_for, generate_reformulations, load_cache, load_fiqa, score_query
from reformulation.exceptions import ReformulationParseError
from reformulation.llm.exceptions import LLMRateLimitError, LLMTimeoutError


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def sleep(self, s):
        self.sleeps.append(round(s, 3))
        self.now += s

    def clock(self):
        return self.now


class ScriptedReformulator:
    """Returns/raises the scripted outcome for each call, in order."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_variants(self, query, max_variants=5):
        self.calls.append(query)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestGenerateReformulations(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_path = os.path.join(self.tmp.name, "cache.json")
        self.queries = {"q1": "roth ira limit", "q2": "etf vs mutual fund", "q3": "tax on dividends"}
        self.t = FakeClock()

    def tearDown(self):
        self.tmp.cleanup()

    def run_gen(self, reformulator, **kw):
        return generate_reformulations(self.queries, reformulator, cache_path=self.cache_path,
                                       min_interval=13.0, rate_limit_wait=65.0,
                                       sleep=self.t.sleep, clock=self.t.clock, **kw)

    def test_all_queries_cached_and_paced(self):
        stats = self.run_gen(ScriptedReformulator([["a"], ["b"], ["c"]]))
        self.assertEqual(stats["new"], 3)
        self.assertEqual(stats["remaining"], 0)
        self.assertEqual(load_cache(self.cache_path)["reformulations"], {"q1": ["a"], "q2": ["b"], "q3": ["c"]})
        self.assertEqual(self.t.sleeps, [13.0, 13.0])     # no sleep before the first call

    def test_resume_skips_cached_queries(self):
        self.run_gen(ScriptedReformulator([["a"]]), limit=1)
        fake = ScriptedReformulator([["b"], ["c"]])
        stats = self.run_gen(fake)
        self.assertEqual(fake.calls, ["etf vs mutual fund", "tax on dividends"])
        self.assertEqual(stats["already_cached"], 1)
        self.assertEqual(stats["total_cached"], 3)

    def test_rate_limit_waits_and_retries_same_query(self):
        fake = ScriptedReformulator([["a"], LLMRateLimitError("429"), ["b"], ["c"]])
        stats = self.run_gen(fake)
        self.assertEqual(fake.calls, ["roth ira limit", "etf vs mutual fund", "etf vs mutual fund", "tax on dividends"])
        self.assertIn(65.0, self.t.sleeps)
        self.assertIsNone(stats["stopped"])
        self.assertEqual(stats["total_cached"], 3)

    def test_repeated_rate_limits_stop_and_keep_progress(self):
        fake = ScriptedReformulator([["a"]] + [LLMRateLimitError("429")] * 3)
        stats = self.run_gen(fake)
        self.assertIn("rate_limit", stats["stopped"])
        self.assertEqual(stats["remaining"], 2)
        self.assertEqual(load_cache(self.cache_path)["reformulations"], {"q1": ["a"]})

    def test_parse_error_cached_as_empty_and_recorded(self):
        stats = self.run_gen(ScriptedReformulator([ReformulationParseError("bad json"), ["b"], ["c"]]))
        cache = load_cache(self.cache_path)
        self.assertEqual(cache["reformulations"]["q1"], [])
        self.assertIn("bad json", cache["parse_errors"]["q1"])
        self.assertEqual(stats["parse_errors"], 1)

    def test_transient_error_skips_query_without_caching(self):
        stats = self.run_gen(ScriptedReformulator([LLMTimeoutError("t"), ["b"], ["c"]]))
        self.assertNotIn("q1", load_cache(self.cache_path)["reformulations"])
        self.assertEqual(stats["remaining"], 1)
        self.assertIsNone(stats["stopped"])

    def test_many_consecutive_errors_stop(self):
        stats = generate_reformulations(self.queries, ScriptedReformulator([LLMTimeoutError("t")] * 3),
                                        cache_path=self.cache_path, max_consecutive_errors=2,
                                        sleep=self.t.sleep, clock=self.t.clock)
        self.assertIn("errors", stats["stopped"])


class TestLoadFiqaAndScoring(unittest.TestCase):
    def test_cache_path_is_per_model(self):
        self.assertTrue(cache_path_for("gemini-3.5-flash-lite").endswith("beir_fiqa_reformulations_gemini-3.5-flash-lite.json"))
        self.assertTrue(cache_path_for("models/qwen2.5:7b").endswith("beir_fiqa_reformulations_qwen2.5_7b.json"))
        self.assertNotEqual(cache_path_for("a"), cache_path_for("b"))

    def test_load_fiqa_filters_to_split(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "qrels"))
            with open(os.path.join(d, "qrels", "test.tsv"), "w") as f:
                f.write("query-id\tcorpus-id\tscore\n8\t100\t1\n8\t101\t1\n9\t102\t0\n")
            with open(os.path.join(d, "queries.jsonl"), "w") as f:
                for qid, text in (("8", "how to save tax"), ("9", "ignored"), ("10", "train only")):
                    f.write(json.dumps({"_id": qid, "text": text}) + "\n")
            with open(os.path.join(d, "corpus.jsonl"), "w") as f:
                f.write(json.dumps({"_id": "100", "title": "", "text": "use a 401k"}) + "\n")
            chunks, queries, qrels = load_fiqa(d)
        self.assertEqual(queries, {"8": "how to save tax"})          # q9 has only a zero label
        self.assertEqual(qrels, {"8": {"100": 1, "101": 1}})
        self.assertEqual(chunks[0]["chunk_id"], "100")
        self.assertEqual(chunks[0]["text"], "use a 401k")

    def test_score_query(self):
        m = score_query(["x", "100", "y"], {"100": 1, "101": 1})
        self.assertAlmostEqual(m["recall@10"], 0.5)
        self.assertAlmostEqual(m["recall@20"], 0.5)
        self.assertAlmostEqual(m["mrr"], 0.5)
        self.assertGreater(m["ndcg@10"], 0)


if __name__ == "__main__":
    unittest.main()
