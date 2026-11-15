"""Dataset boundary tests: hash checks, blind-test freezing, human-label provenance."""

import json
import os
import tempfile
import unittest

from evaluation.datasets.splits import (
    BlindTestNotFrozenError, DatasetIntegrityError, load_split, sha256_file, validate_human_labels,
)


class TestSplits(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.qpath = os.path.join(self.tmp, "q.json")
        self.lpath = os.path.join(self.tmp, "l.json")
        with open(self.qpath, "w", encoding="utf-8") as f:
            json.dump({"queries": [{"query_id": "BT001", "query": "pet policy", "category": "unanswerable"},
                                   {"query_id": "BT002", "query": "curfew", "category": "x"}]}, f)
        self.labels = {
            "label_source": "human", "annotators": [{"annotator_id": "A1"}, {"annotator_id": "A2"}],
            "queries": {
                "BT001": {"answerability": "unanswerable", "judgments": []},
                "BT002": {"answerability": "fully_answerable", "judgments": [
                    {"chunk_id": "C1", "grade": 3, "annotator_id": "A1", "annotated_at": "2026-11-20"},
                    {"chunk_id": "C1", "grade": 2, "annotator_id": "A2", "annotated_at": "2026-11-20"},
                    {"chunk_id": "C2", "grade": 1, "annotator_id": "A1", "annotated_at": "2026-11-20"}]}}}

    def manifest(self, frozen, labels_hash=None, qhash=None):
        m = {"splits": {
            "development": {"queries_file": self.qpath, "queries_sha256": qhash or sha256_file(self.qpath)},
            "validation_contaminated": {"queries_file": self.qpath, "queries_sha256": sha256_file(self.qpath)},
            "blind_test": {"queries_file": self.qpath, "queries_sha256": sha256_file(self.qpath),
                           "labels_file": self.lpath, "labels_sha256": labels_hash, "frozen": frozen}}}
        path = os.path.join(self.tmp, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(m, f)
        return path

    def write_labels(self, labels=None):
        with open(self.lpath, "w", encoding="utf-8") as f:
            json.dump(labels or self.labels, f)

    def test_query_file_hash_checked(self):
        with self.assertRaises(DatasetIntegrityError):
            load_split("development", self.manifest(False, qhash="0" * 64))

    def test_blind_test_refused_until_frozen(self):
        self.write_labels()
        with self.assertRaises(BlindTestNotFrozenError):
            load_split("blind_test", self.manifest(False))

    def test_frozen_blind_test_loads_and_merges_annotators(self):
        self.write_labels()
        qs, _ = load_split("blind_test", self.manifest(True, sha256_file(self.lpath)))
        by = {q["query_id"]: q for q in qs}
        self.assertEqual(by["BT001"]["relevant_chunks"], [])
        self.assertEqual(by["BT001"]["answerability"], "unanswerable")
        self.assertEqual(by["BT002"]["graded_relevance"], {"C1": 2, "C2": 1})  # median rounded down
        self.assertEqual(by["BT002"]["relevant_chunks"], ["C1"])

    def test_modified_labels_after_freeze_detected(self):
        self.write_labels()
        path = self.manifest(True, sha256_file(self.lpath))
        self.labels["queries"]["BT002"]["judgments"][0]["grade"] = 0
        self.write_labels()
        with self.assertRaises(DatasetIntegrityError):
            load_split("blind_test", path)

    def test_non_human_or_incomplete_labels_rejected(self):
        bad = dict(self.labels, label_source="llm_judge")
        with self.assertRaises(DatasetIntegrityError):
            validate_human_labels(bad, ["BT001", "BT002"])
        with self.assertRaises(DatasetIntegrityError):
            validate_human_labels(self.labels, ["BT001", "BT002", "BT003"])
        no_ann = json.loads(json.dumps(self.labels))
        no_ann["queries"]["BT002"]["judgments"][0]["annotator_id"] = "UNKNOWN"
        with self.assertRaises(DatasetIntegrityError):
            validate_human_labels(no_ann, ["BT001", "BT002"])

    def test_repository_manifest_matches_files(self):
        for split in ("development", "validation_contaminated"):
            qs, info = load_split(split)
            self.assertTrue(qs)
        with self.assertRaises(BlindTestNotFrozenError):
            load_split("blind_test")

    def test_blind_test_file_meets_coverage_requirements(self):
        from collections import Counter
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        data = json.load(open(os.path.join(root, "evaluation", "datasets", "blind_test_v1.json"), encoding="utf-8"))
        cats = Counter(q["category"] for q in data["queries"])
        self.assertGreaterEqual(cats["unanswerable"], 30)
        self.assertGreater(cats["typo_heavy"], 0)
        self.assertGreater(cats["multi_intent"], 0)
        self.assertTrue(any(c.startswith("precise_") for c in cats))
        self.assertTrue(all(q["label_status"] == "unlabelled" and q["relevant_chunks"] is None for q in data["queries"]))


if __name__ == "__main__":
    unittest.main()
