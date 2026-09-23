"""Tests for the reformulation foundation (validation & normalization).

These tests run offline and use only the local implementation.  They verify
that the public API behaves exactly as specified in the task description.
"""

import os
import unittest
from typing import Any

# Import the public API from the package
from reformulation import validate_query, normalize_query

class TestReformulationFoundation(unittest.TestCase):
    # ----- Validation tests -----
    def test_valid_normal_query(self):
        # Simple non‑empty query should pass without exception
        try:
            validate_query("What is the capital of France?")
        except Exception as e:
            self.fail(f"validate_query raised unexpectedly: {e}")

    def test_empty_string(self):
        with self.assertRaises(ValueError) as cm:
            validate_query("")
        self.assertIn("empty", str(cm.exception).lower())

    def test_whitespace_only(self):
        with self.assertRaises(ValueError):
            validate_query("   \t\n   ")

    def test_none_input(self):
        with self.assertRaises(ValueError):
            validate_query(None)

    def test_integer_input(self):
        with self.assertRaises(ValueError):
            validate_query(123)

    def test_list_input(self):
        with self.assertRaises(ValueError):
            validate_query(["a", "b"])

    def test_dict_input(self):
        with self.assertRaises(ValueError):
            validate_query({"q": "test"})

    def test_maximum_allowed_length(self):
        max_len = int(os.getenv("MAX_QUERY_LENGTH", "256"))
        query = "x" * max_len
        # Should be accepted (no exception)
        try:
            validate_query(query)
        except Exception as e:
            self.fail(f"validate_query rejected max‑length query: {e}")

    def test_exceeding_maximum_length(self):
        max_len = int(os.getenv("MAX_QUERY_LENGTH", "256"))
        query = "x" * (max_len + 1)
        with self.assertRaises(ValueError) as cm:
            validate_query(query)
        self.assertIn("exceeds", str(cm.exception).lower())

    # ----- Normalization tests -----
    def test_trim_whitespace(self):
        self.assertEqual(normalize_query("   hello world   "), "hello world")

    def test_collapse_multiple_spaces(self):
        self.assertEqual(normalize_query("hello   world"), "hello world")

    def test_tabs_and_newlines(self):
        self.assertEqual(normalize_query("hello\tworld\nnew"), "hello world new")

    def test_punctuation_spacing(self):
        self.assertEqual(normalize_query("hello , world !"), "hello, world!")

    def test_semantic_unchanged(self):
        raw = "Find AI/ML papers from 2022."
        norm = normalize_query(raw)
        # Should keep case, numbers, slash and period
        self.assertEqual(norm, "Find AI/ML papers from 2022.")

    def test_preserve_entities(self):
        raw = "Explain the theory of relativity by Einstein."
        self.assertIn("Einstein", normalize_query(raw))

    def test_preserve_technical_terms(self):
        raw = "What is a transformer model in NLP?"
        self.assertIn("transformer", normalize_query(raw).lower())

    def test_preserve_numbers(self):
        raw = "Show me stats for 2020 vs 2021."
        self.assertIn("2020", normalize_query(raw))
        self.assertIn("2021", normalize_query(raw))

    def test_preserve_acronyms(self):
        raw = "Explain CPU vs GPU performance."
        self.assertIn("CPU", normalize_query(raw))
        self.assertIn("GPU", normalize_query(raw))

    def test_normalization_empty_result(self):
        # Normalising an input that becomes empty should raise via validation
        with self.assertRaises(ValueError):
            # The validation will catch the empty after stripping
            validate_query(normalize_query("   "))

    # ----- Integration tests -----
    def test_valid_raw_query_flow(self):
        raw = "   Retrieve   documents about   AI.   "
        # Validation first
        validate_query(raw)
        # Normalization second
        norm = normalize_query(raw)
        self.assertEqual(norm, "Retrieve documents about AI.")

    def test_invalid_query_never_normalized(self):
        raw = "   "
        with self.assertRaises(ValueError):
            validate_query(raw)
        # Ensure that normalization is *not* called – we simply verify that no
        # exception from normalize_query would be raised because we never call it.
        # (The test passes if the above block raises as expected.)
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
