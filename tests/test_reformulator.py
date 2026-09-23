import json
import unittest
from unittest.mock import MagicMock, patch

from reformulation import validate_query, normalize_query
from reformulation.reformulator import QueryReformulator
from reformulation.exceptions import ReformulationParseError
from reformulation.llm.client import LLMClient

class TestQueryReformulator(unittest.TestCase):
    def setUp(self):
        # Create a mock LLMClient
        self.mock_client = MagicMock(spec=LLMClient)
        self.reformulator = QueryReformulator(self.mock_client)

    def test_basic_successful_reformulation(self):
        # Mock LLM response JSON
        response = json.dumps({"reformulations": ["alternative one", "alternative two"]})
        self.mock_client.generate.return_value = response
        result = self.reformulator.generate_variants("attendance policy", max_variants=5)
        self.assertEqual(result, ["alternative one", "alternative two"])
        # ensure prompt contains normalized query
        prompt_sent = self.mock_client.generate.call_args[0][0]
        self.assertIn("attendance policy", prompt_sent)

    def test_validation_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.reformulator.generate_variants(123)
        with self.assertRaises(ValueError):
            self.reformulator.generate_variants("")
        with self.assertRaises(ValueError):
            self.reformulator.generate_variants("   ")

    def test_normalization_used_in_prompt(self):        # input with extra whitespace
        self.mock_client.generate.return_value = json.dumps({"reformulations": []})
        self.reformulator.generate_variants("  What   is   leave policy?  ")
        prompt = self.mock_client.generate.call_args[0][0]
        self.assertIn("What is leave policy?", prompt)

    def test_max_variants_limit(self):
        response = json.dumps({"reformulations": ["a", "b", "c", "d", "e"]})
        self.mock_client.generate.return_value = response
        result = self.reformulator.generate_variants("policy", max_variants=3)
        self.assertEqual(len(result), 3)
        self.assertEqual(result, ["a", "b", "c"])

    def test_duplicate_removal(self):
        response = json.dumps({"reformulations": ["same", "same", "different", "same"]})
        self.mock_client.generate.return_value = response
        result = self.reformulator.generate_variants("same query")
        self.assertEqual(result, ["same", "different"])

    def test_duplicate_removal_multiple(self):
        response = json.dumps({"reformulations": ["A", "A", "B", "A", "C", "B"]})
        self.mock_client.generate.return_value = response
        result = self.reformulator.generate_variants("query")
        self.assertEqual(result, ["A", "B", "C"])

    def test_original_query_duplicate_removed(self):
        normalized = normalize_query("query original")
        response = json.dumps({"reformulations": [normalized, "other"]})
        self.mock_client.generate.return_value = response
        result = self.reformulator.generate_variants("query original")
        self.assertEqual(result, ["other"])

    def test_empty_results(self):
        response = json.dumps({"reformulations": []})
        self.mock_client.generate.return_value = response
        result = self.reformulator.generate_variants("any")
        self.assertEqual(result, [])

    def test_malformed_json(self):
        self.mock_client.generate.return_value = "not json"
        with self.assertRaises(ReformulationParseError):
            self.reformulator.generate_variants("test")

    def test_missing_field(self):
        self.mock_client.generate.return_value = json.dumps({"queries": []})
        with self.assertRaises(ReformulationParseError):
            self.reformulator.generate_variants("test")

    def test_wrong_field_type(self):
        self.mock_client.generate.return_value = json.dumps({"reformulations": "not a list"})
        with self.assertRaises(ReformulationParseError):
            self.reformulator.generate_variants("test")

    def test_invalid_item_type(self):
        self.mock_client.generate.return_value = json.dumps({"reformulations": ["valid", 123]})
        with self.assertRaises(ReformulationParseError):
            self.reformulator.generate_variants("test")

    def test_whitespace_items_removed(self):
        self.mock_client.generate.return_value = json.dumps({"reformulations": ["", "   ", "valid"]})
        result = self.reformulator.generate_variants("test")
        self.assertEqual(result, ["valid"])

    def test_invalid_max_variants(self):
        with self.assertRaises(ValueError):
            self.reformulator.generate_variants("test", max_variants=0)
        with self.assertRaises(ValueError):
            self.reformulator.generate_variants("test", max_variants=-2)

    def test_llm_error_propagation(self):
        from reformulation.llm.exceptions import LLMAuthError
        self.mock_client.generate.side_effect = LLMAuthError("auth failed")
        with self.assertRaises(LLMAuthError):
            self.reformulator.generate_variants("test")

    def test_deterministic_output(self):
        response = json.dumps({"reformulations": ["a", "b"]})
        self.mock_client.generate.return_value = response
        first = self.reformulator.generate_variants("test")
        second = self.reformulator.generate_variants("test")
        self.assertEqual(first, second)

if __name__ == "__main__":
    unittest.main()
