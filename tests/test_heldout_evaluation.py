import json
import os
import tempfile
import unittest

from evaluation.build_reformulation_cache import cache_path_for
from evaluation.run_heldout_evaluation import COMPARISONS, SYSTEMS, load_reformulations
from evaluation.schema import GoldenQuery, QueryCategory

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
HELDOUT_PATH = os.path.join(ROOT, "evaluation", "heldout_queries.json")
GOLDEN_PATH = os.path.join(ROOT, "evaluation", "golden_queries.json")
CHUNKS_PATH = os.path.join(ROOT, "Data", "chunks", "chunks.jsonl")


class TestHeldoutDataset(unittest.TestCase):
    """The held-out set must stay valid against the corpus and separate from the tuning set."""

    @classmethod
    def setUpClass(cls):
        with open(HELDOUT_PATH, encoding="utf-8") as f:
            cls.heldout = json.load(f)
        with open(GOLDEN_PATH, encoding="utf-8") as f:
            cls.golden = json.load(f)
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            cls.chunk_ids = {json.loads(line)["chunk_id"] for line in f if line.strip()}

    def test_ids_unique_and_disjoint_from_tuning_set(self):
        ids = [q["query_id"] for q in self.heldout]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(set(ids) & {q["query_id"] for q in self.golden})
        tuning_texts = {q["query"].strip().lower() for q in self.golden}
        self.assertFalse({q["query"].strip().lower() for q in self.heldout} & tuning_texts)

    def test_labels_point_at_real_chunks(self):
        for q in self.heldout:
            for cid, grade in q["graded_relevance"].items():
                self.assertIn(cid, self.chunk_ids, q["query_id"])
                self.assertIn(grade, (1, 2, 3), q["query_id"])
            # relevant_chunks = the chunks graded 2 or 3
            self.assertEqual(set(q["relevant_chunks"]),
                             {c for c, g in q["graded_relevance"].items() if g >= 2}, q["query_id"])

    def test_answerable_flag_matches_labels(self):
        for q in self.heldout:
            self.assertEqual(q["answerable"], bool(q["relevant_chunks"]), q["query_id"])
        self.assertGreaterEqual(sum(q["answerable"] for q in self.heldout), 90)

    def test_categories_and_schema(self):
        valid = {c.value for c in QueryCategory}
        for q in self.heldout:
            self.assertIn(q["category"], valid)
            self.assertEqual(q["split"], "test")
            if q["answerable"]:
                GoldenQuery.from_dict(q)  # loads with the shared schema


class TestHeldoutHelpers(unittest.TestCase):
    def test_load_reformulations_both_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            plain, wrapped = os.path.join(tmp, "a.json"), os.path.join(tmp, "b.json")
            with open(plain, "w", encoding="utf-8") as f:
                json.dump({"Q1": ["x"]}, f)
            with open(wrapped, "w", encoding="utf-8") as f:
                json.dump({"reformulations": {"Q1": ["y"]}, "parse_errors": {}}, f)
            self.assertEqual(load_reformulations(plain), {"Q1": ["x"]})
            self.assertEqual(load_reformulations(wrapped), {"Q1": ["y"]})

    def test_cache_path_per_queries_file_and_model(self):
        path = cache_path_for("evaluation/heldout_queries.json", "models/gemini-3.8-flash")
        self.assertTrue(path.endswith(os.path.join("cache", "heldout_queries_reformulations_gemini-3.8-flash.json")))

    def test_comparisons_reference_known_systems(self):
        for a, b, _ in COMPARISONS:
            self.assertIn(a, SYSTEMS)
            self.assertIn(b, SYSTEMS)


if __name__ == "__main__":
    unittest.main()
