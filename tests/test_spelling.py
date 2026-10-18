import json
import tempfile
import unittest
from pathlib import Path

from reformulation.normalizer import normalize_query
from reformulation.spelling import CorpusSpellCorrector, _edit_distance, build_vocabulary


class TestEditDistance(unittest.TestCase):
    def test_basic_edits(self):
        self.assertEqual(_edit_distance("libary", "library", 1), 1)      # insertion
        self.assertEqual(_edit_distance("hostle", "hostel", 1), 1)       # adjacent swap
        self.assertEqual(_edit_distance("tiket", "ticket", 1), 1)
        self.assertEqual(_edit_distance("fees", "fees", 1), 0)

    def test_limit_cutoff(self):
        self.assertEqual(_edit_distance("revalution", "resolution", 1), 2)
        self.assertEqual(_edit_distance("abc", "abcdefgh", 1), 2)


class TestCorpusSpellCorrector(unittest.TestCase):
    def setUp(self):
        self.c = CorpusSpellCorrector({
            "library": 30, "examination": 20, "ticket": 5, "withdrawal": 7,
            "fees": 40, "feet": 2, "resolution": 20, "hostel": 12,
            "form": 10, "farm": 10,
        })

    def test_corrects_one_edit_typos(self):
        self.assertEqual(self.c.correct("libary"), "library")
        self.assertEqual(self.c.correct("examinaton"), "examination")
        self.assertEqual(self.c.correct("withdrawl"), "withdrawal")
        self.assertEqual(self.c.correct("hostle"), "hostel")

    def test_known_short_and_non_alpha_words_untouched(self):
        self.assertIsNone(self.c.correct("library"))
        self.assertIsNone(self.c.correct("fes"))        # shorter than 4 letters
        self.assertIsNone(self.c.correct("cse301"))
        self.assertIsNone(self.c.correct("802"))

    def test_two_edit_guess_rejected(self):
        self.assertIsNone(self.c.correct("revalution"))

    def test_first_letter_must_match(self):
        self.assertIsNone(self.c.correct("gees"))

    def test_ambiguous_tie_not_corrected(self):
        # "firm" is one edit from both "form" and "farm", which are equally common.
        self.assertIsNone(self.c.correct("firm"))

    def test_clear_frequency_winner_corrected(self):
        # "feez" is one edit from "fees" (40) and "feet" (2): the far more common word wins.
        self.assertEqual(self.c.correct("feez"), "fees")

    def test_normalizer_leaves_acronyms_and_names(self):
        self.assertEqual(normalize_query("Dr Feez office", spell_corrector=self.c), "Dr Feez office")
        self.assertEqual(normalize_query("LIBARY", spell_corrector=self.c), "Library")  # all-caps query is noisy
        self.assertEqual(normalize_query("libary fees", spell_corrector=self.c), "Library fees")

    def test_from_chunks_file_uses_document_text_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chunks.jsonl"
            rows = [{"title": "Hostel Rules", "section": "1. Allotment", "text": "Rooms are allotted yearly.",
                     "keywords": ["revaluation"]}]
            path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            c = CorpusSpellCorrector.from_chunks_file(path)
        self.assertIn("allotment", c)
        self.assertNotIn("revaluation", c)  # generator metadata is not vocabulary

    def test_build_vocabulary(self):
        v = build_vocabulary(["Fee fee FEES", "a an the"])
        self.assertEqual(v["fee"], 2)
        self.assertEqual(v["fees"], 1)
        self.assertNotIn("an", v)


if __name__ == "__main__":
    unittest.main()
