import json
import os
import tempfile
import unittest

from evaluation.prepare_msmarco_sample import bm25_pools, sample_query_ids, tokenize
from retrieval.bm25_retriever import BM25Retriever
from retrieval.bm25_retriever import tokenize as bm25_tokenize

PASSAGES = [
    ("p1", "Tax rules", "capital gains tax applies when you sell shares after a year"),
    ("p2", "", "the cost of a passport renewal is about 130 dollars"),
    ("p3", "Passports", "renewing a passport by mail takes six to eight weeks"),
    ("p4", "", "shares bought and sold within a year are taxed as income"),
    ("p5", "Cooking", "boil pasta for ten minutes in salted water"),
    ("p6", "", "passport photos must be two by two inches"),
]
QUERIES = {"q1": "how long does passport renewal take", "q2": "tax on selling shares"}


class TestStreamingBM25(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = os.path.join(self.tmp.name, "corpus.jsonl")
        with open(self.corpus, "w", encoding="utf-8") as f:
            for pid, title, text in PASSAGES:
                f.write(json.dumps({"_id": pid, "title": title, "text": text}) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_tokenizer_matches_bm25_retriever(self):
        text = "Renewing PASSPORTS, fees & taxes in 2024"
        self.assertEqual(tokenize(text), bm25_tokenize(text))

    def test_pools_match_in_process_bm25_ranking(self):
        pools = bm25_pools(self.corpus, QUERIES, depth=3, log_every=10**9)
        chunks = [{"chunk_id": pid, "document_id": pid, "title": title, "section": None, "text": text,
                   "chunk_index": 0} for pid, title, text in PASSAGES]
        bm25 = BM25Retriever(chunks=chunks)
        for qid, query in QUERIES.items():
            expected = bm25.retrieve(query, match_count=3)
            self.assertEqual([pid for _, pid in pools[qid]], [c.chunk_id for c in expected], qid)
            for (score, _), chunk in zip(pools[qid], expected):
                self.assertAlmostEqual(score, chunk.score, places=9)

    def test_depth_limits_pool(self):
        pools = bm25_pools(self.corpus, QUERIES, depth=1, log_every=10**9)
        self.assertEqual([len(p) for p in pools.values()], [1, 1])

    def test_sampling_is_seeded_and_sorted(self):
        qrels = {f"q{i}": {"p": 1} for i in range(50)}
        a, b = sample_query_ids(qrels, 10, seed=0), sample_query_ids(qrels, 10, seed=0)
        self.assertEqual(a, b)
        self.assertEqual(a, sorted(a))
        self.assertNotEqual(a, sample_query_ids(qrels, 10, seed=1))
        self.assertEqual(len(sample_query_ids(qrels, 999, seed=0)), 50)


if __name__ == "__main__":
    unittest.main()
