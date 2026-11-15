"""labels_from_csv must never produce labels from incomplete or unattributed annotation rows."""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(ROOT, "evaluation", "datasets", "labels_from_csv.py")


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


class TestLabelsFromCsv(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.ann = os.path.join(self.d, "annotators.json")
        json.dump([{"annotator_id": "A1", "role": "registrar"}], open(self.ann, "w"))
        self.pool = os.path.join(self.d, "pool.csv")
        self.ans = os.path.join(self.d, "ans.csv")
        self.out = os.path.join(self.d, "labels.json")
        write_csv(self.ans, [{"query_id": "BT001", "answerability": "unanswerable", "annotator_id": "A1",
                              "annotated_at": "2026-11-20"}])

    def run_script(self):
        return subprocess.run([sys.executable, SCRIPT, "--annotators", self.ann, "--pool", self.pool,
                               "--answerability", self.ans, "--out", self.out], capture_output=True, text=True)

    def test_blank_grade_refused(self):
        write_csv(self.pool, [{"query_id": "BT001", "chunk_id": "C1", "grade": "", "annotator_id": "A1",
                               "annotated_at": "2026-11-20", "notes": ""}])
        r = self.run_script()
        self.assertEqual(r.returncode, 1)
        self.assertFalse(os.path.exists(self.out))

    def test_unknown_annotator_refused(self):
        write_csv(self.pool, [{"query_id": "BT001", "chunk_id": "C1", "grade": "0", "annotator_id": "LLM",
                               "annotated_at": "2026-11-20", "notes": ""}])
        self.assertEqual(self.run_script().returncode, 1)

    def test_complete_rows_written_as_human(self):
        write_csv(self.pool, [{"query_id": "BT001", "chunk_id": "C1", "grade": "0", "annotator_id": "A1",
                               "annotated_at": "2026-11-20", "notes": ""}])
        r = self.run_script()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        labels = json.load(open(self.out))
        self.assertEqual(labels["label_source"], "human")
        self.assertEqual(labels["queries"]["BT001"]["judgments"][0]["grade"], 0)


if __name__ == "__main__":
    unittest.main()
