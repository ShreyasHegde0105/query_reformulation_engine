import json
import os
import tempfile
import unittest

from embedding.mapper import DocumentMapper
from retrieval.bm25_retriever import BM25Retriever, tokenize
from retrieval.models import RetrievedChunk


def _chunk(cid, text, title="Doc", section=None, dept="Academic Affairs"):
    return {
        "chunk_id": cid,
        "document_id": cid.split("_")[0],
        "title": title,
        "section": section,
        "text": text,
        "chunk_index": 0,
        "token_count": len(text.split()),
        "department": dept,
    }


CHUNKS = [
    _chunk("FIN_CHUNK_001", "Tuition fees are payable each semester before the deadline.", "Fee Policy", "2. FEES", "Finance"),
    _chunk("EXAM_CHUNK_001", "Revaluation of answer scripts costs 500 rupees per paper.", "Revaluation Guidelines", "3. REVALUATION", "Examinations"),
    _chunk("ACAD_CHUNK_001", "Every student must attend classes each semester.", "Attendance Policy", "1. PURPOSE"),
    _chunk("ACAD_CHUNK_002", "Students below 75 percent attendance are detained.", "Attendance Policy", "2. SHORTAGE"),
    _chunk("CRS_CHUNK_001", "Prerequisites: data structures and discrete mathematics.", "CSE301 Operating Systems", "1. COURSE INFO"),
]


class TestTokenize(unittest.TestCase):
    def test_lowercases_drops_stop_words_and_stems_plurals(self):
        self.assertEqual(tokenize("The Fees of the Students"), ["fee", "student"])

    def test_keeps_alphanumeric_identifiers(self):
        self.assertIn("cse301", tokenize("CSE301 syllabus"))

    def test_short_words_and_numbers_unchanged(self):
        self.assertEqual(tokenize("bus 75 gas"), ["bus", "75", "gas"])


class TestBM25Retriever(unittest.TestCase):
    def setUp(self):
        self.retriever = BM25Retriever(chunks=CHUNKS)

    def test_returns_retrieved_chunks_with_keyword_type(self):
        results = self.retriever.retrieve("revaluation of answer scripts")
        self.assertTrue(results)
        self.assertIsInstance(results[0], RetrievedChunk)
        self.assertEqual(results[0].retriever_type, "keyword")
        self.assertEqual(results[0].chunk_id, "EXAM_CHUNK_001")
        self.assertEqual(results[0].domain, "Examinations")

    def test_results_sorted_by_score_descending(self):
        scores = [c.score for c in self.retriever.retrieve("semester attendance students")]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_or_semantics_partial_match_is_returned(self):
        # "hostel" matches nothing; "fees" still matches (Postgres AND semantics would return nothing).
        ids = [c.chunk_id for c in self.retriever.retrieve("hostel fees")]
        self.assertEqual(ids, ["FIN_CHUNK_001"])

    def test_no_overlap_returns_empty(self):
        self.assertEqual(self.retriever.retrieve("hostel wifi"), [])

    def test_rare_term_outranks_common_term(self):
        # "semester" occurs in two chunks, "detained" in one: IDF must favour the rare term.
        results = self.retriever.retrieve("semester detained")
        self.assertEqual(results[0].chunk_id, "ACAD_CHUNK_002")

    def test_title_and_section_are_indexed(self):
        ids = [c.chunk_id for c in self.retriever.retrieve("CSE301")]
        self.assertEqual(ids, ["CRS_CHUNK_001"])

    def test_fields_restrict_indexed_text(self):
        text_only = BM25Retriever(chunks=CHUNKS, fields=("text",))
        self.assertEqual(text_only.retrieve("CSE301"), [])

    def test_match_count_limits_results(self):
        self.assertEqual(len(self.retriever.retrieve("semester students attendance fees", match_count=2)), 2)

    def test_filter_domain(self):
        results = self.retriever.retrieve("semester", filter_domain="Finance")
        self.assertEqual([c.chunk_id for c in results], ["FIN_CHUNK_001"])

    def test_deterministic_tie_break_by_chunk_id(self):
        tied = BM25Retriever(chunks=[_chunk("B_CHUNK_001", "library"), _chunk("A_CHUNK_001", "library")])
        self.assertEqual([c.chunk_id for c in tied.retrieve("library")], ["A_CHUNK_001", "B_CHUNK_001"])

    def test_invalid_arguments(self):
        for bad in ("", "   ", None, 5):
            with self.assertRaises(ValueError):
                self.retriever.retrieve(bad)
        for bad in (0, 101, True, "10"):
            with self.assertRaises(ValueError):
                self.retriever.retrieve("fees", match_count=bad)
        with self.assertRaises(ValueError):
            self.retriever.retrieve("fees", filter_domain=3)
        with self.assertRaises(ValueError):
            BM25Retriever(chunks=CHUNKS, b=1.5)

    def test_loads_chunks_from_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "chunks.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for c in CHUNKS:
                    f.write(json.dumps(c) + "\n")
            retriever = BM25Retriever(chunks_path=path)
            self.assertEqual(retriever.retrieve("revaluation")[0].chunk_id, "EXAM_CHUNK_001")

    def test_missing_chunks_file(self):
        with self.assertRaises(FileNotFoundError):
            BM25Retriever(chunks_path="does/not/exist.jsonl")


class TestPassageText(unittest.TestCase):
    def test_contextual_prepends_title_and_section(self):
        c = _chunk("X_CHUNK_001", "Body text.", "Fee Policy", "2. FEES")
        self.assertEqual(DocumentMapper.passage_text(c), "Fee Policy\n2. FEES\n\nBody text.")

    def test_contextual_skips_missing_section(self):
        c = _chunk("X_CHUNK_001", "Body text.", "Library FAQ", None)
        self.assertEqual(DocumentMapper.passage_text(c), "Library FAQ\n\nBody text.")

    def test_plain_returns_text_unchanged(self):
        c = _chunk("X_CHUNK_001", "Body text.", "Fee Policy", "2. FEES")
        self.assertEqual(DocumentMapper.passage_text(c, contextual=False), "Body text.")


if __name__ == "__main__":
    unittest.main()
