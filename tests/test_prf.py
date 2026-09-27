import unittest

from retrieval.bm25_retriever import BM25Retriever
from retrieval.prf import RM3Retriever, rm3_term_weights


def _c(cid, text):
    return {"chunk_id": cid, "document_id": cid, "title": "", "section": None, "text": text, "department": "X"}


CHUNKS = [
    _c("D1", "roth ira contribution limit retirement account tax"),
    _c("D2", "ira contribution rules retirement savings tax deduction"),
    _c("D3", "retirement account withdrawal penalty early distribution"),
    _c("D4", "stock market dividend yield investing"),
    _c("D5", "credit card interest rate debt"),
]


class TestRM3(unittest.TestCase):
    def setUp(self):
        self.bm25 = BM25Retriever(chunks=CHUNKS)

    def test_weights_form_distribution_and_keep_query_terms(self):
        w = rm3_term_weights("ira contribution", self.bm25, fb_docs=2, fb_terms=5)
        self.assertAlmostEqual(sum(w.values()), 1.0)
        self.assertIn("ira", w)
        self.assertIn("contribution", w)
        self.assertIn("retirement", w)            # expansion term from feedback docs
        self.assertNotIn("dividend", w)           # never in the feedback docs

    def test_original_weight_one_is_plain_query(self):
        w = rm3_term_weights("ira contribution", self.bm25, original_query_weight=1.0)
        self.assertEqual({t: v for t, v in w.items() if v > 0}, {"ira": 0.5, "contribution": 0.5})

    def test_expansion_finds_related_doc_without_query_terms(self):
        plain = [c.chunk_id for c in self.bm25.retrieve("roth ira", match_count=5)]
        expanded = [c.chunk_id for c in RM3Retriever(self.bm25, fb_docs=2, fb_terms=10).retrieve("roth ira", match_count=5)]
        self.assertNotIn("D3", plain)
        self.assertIn("D3", expanded)             # shares "retirement account" with the feedback docs
        self.assertEqual(expanded[0], "D1")

    def test_no_match_and_stopword_queries(self):
        self.assertEqual(rm3_term_weights("the of and", self.bm25), {})
        self.assertEqual(RM3Retriever(self.bm25).retrieve("the of and"), [])
        self.assertEqual(rm3_term_weights("zzz", self.bm25), {"zzz": 1.0})

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            rm3_term_weights("ira", self.bm25, fb_docs=0)
        with self.assertRaises(ValueError):
            rm3_term_weights("ira", self.bm25, original_query_weight=1.5)
        with self.assertRaises(ValueError):
            RM3Retriever(self.bm25).retrieve("  ")

    def test_weighted_retrieval_and_doc_terms(self):
        self.assertEqual(self.bm25.doc_terms("D4")["dividend"], 1)
        heavy = self.bm25.retrieve_weighted({"tax": 1.0, "penalty": 5.0}, match_count=1)
        self.assertEqual(heavy[0].chunk_id, "D3")


if __name__ == "__main__":
    unittest.main()
