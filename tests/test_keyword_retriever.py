from __future__ import annotations

import inspect
import unittest
from unittest.mock import MagicMock

from embedding.client import SupabaseRestClient
from retrieval.keyword_retriever import KeywordRetriever
from retrieval.models import RetrievedChunk


class TestKeywordRetriever(unittest.TestCase):
    """Unit tests for KeywordRetriever using mocked Supabase client."""

    def setUp(self) -> None:
        self.mock_client = MagicMock(spec=SupabaseRestClient)
        self.sample_row = {
            "chunk_id": "POL-ATTEND_CHUNK_001",
            "document_id": "POL-ATTEND",
            "title": "Comprehensive Academic Attendance Policy",
            "domain": "Academic Affairs",
            "chunk_index": 0,
            "chunk_text": "Students must maintain a minimum of 75.0% attendance across all registered courses.",
            "section": "1. Minimum Attendance Threshold",
            "token_count": 55,
            "keyword_score": 0.2435,
            "metadata": {"department": "Academic Affairs", "status": "Active"},
        }
        self.mock_client.rpc.return_value = [self.sample_row]
        self.retriever = KeywordRetriever(client=self.mock_client)

    # -------------------------------------------------------------------------
    # 1. Successful retrieval
    # -------------------------------------------------------------------------
    def test_01_successful_retrieval(self) -> None:
        """Verify that a valid query returns typed RetrievedChunk objects."""
        results = self.retriever.retrieve("attendance policy")
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], RetrievedChunk)
        self.assertEqual(results[0].chunk_id, "POL-ATTEND_CHUNK_001")
        self.assertEqual(results[0].score, 0.2435)
        self.assertEqual(results[0].retriever_type, "keyword")

    # -------------------------------------------------------------------------
    # 2. Empty query
    # -------------------------------------------------------------------------
    def test_02_empty_query(self) -> None:
        """Verify ValueError for empty string."""
        with self.assertRaises(ValueError) as ctx:
            self.retriever.retrieve("")
        self.assertIn("non-empty", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # 3. Whitespace query
    # -------------------------------------------------------------------------
    def test_03_whitespace_query(self) -> None:
        """Verify ValueError for whitespace-only query."""
        with self.assertRaises(ValueError) as ctx:
            self.retriever.retrieve("     \t \n  ")
        self.assertIn("non-empty", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # 4. Invalid match_count
    # -------------------------------------------------------------------------
    def test_04_invalid_match_count(self) -> None:
        """Test 0, negative values, values > 100, boolean, and non-ints."""
        with self.assertRaises(ValueError):
            self.retriever.retrieve("test", match_count=0)
        with self.assertRaises(ValueError):
            self.retriever.retrieve("test", match_count=-5)
        with self.assertRaises(ValueError):
            self.retriever.retrieve("test", match_count=101)
        # Boolean trap (bool is a subclass of int in Python)
        with self.assertRaises(ValueError):
            self.retriever.retrieve("test", match_count=True)
        with self.assertRaises(ValueError):
            self.retriever.retrieve("test", match_count="20")  # type: ignore

    # -------------------------------------------------------------------------
    # 5. RPC payload
    # -------------------------------------------------------------------------
    def test_05_rpc_payload(self) -> None:
        """Verify match_chunks_keyword is called with exactly expected payload."""
        self.retriever.retrieve("hostel fee payment", match_count=15)
        self.mock_client.rpc.assert_called_once_with(
            "match_chunks_keyword",
            {
                "query_text": "hostel fee payment",
                "match_count": 15,
                "filter_domain": None,
            },
        )

    # -------------------------------------------------------------------------
    # 6. Domain filter
    # -------------------------------------------------------------------------
    def test_06_domain_filter_passed(self) -> None:
        """Verify filter_domain is passed correctly in payload."""
        self.retriever.retrieve("examination rules", filter_domain="Examination Cell")
        self.mock_client.rpc.assert_called_once_with(
            "match_chunks_keyword",
            {
                "query_text": "examination rules",
                "match_count": 20,
                "filter_domain": "Examination Cell",
            },
        )

    # -------------------------------------------------------------------------
    # 7. Zero results
    # -------------------------------------------------------------------------
    def test_07_zero_results(self) -> None:
        """RPC returns []. KeywordRetriever should return []."""
        self.mock_client.rpc.return_value = []
        results = self.retriever.retrieve("unmatched query term")
        self.assertEqual(results, [])

    # -------------------------------------------------------------------------
    # 8. Result mapping
    # -------------------------------------------------------------------------
    def test_08_result_mapping(self) -> None:
        """Verify all returned database fields map correctly to RetrievedChunk."""
        results = self.retriever.retrieve("attendance policy")
        chunk = results[0]
        self.assertEqual(chunk.chunk_id, "POL-ATTEND_CHUNK_001")
        self.assertEqual(chunk.document_id, "POL-ATTEND")
        self.assertEqual(chunk.title, "Comprehensive Academic Attendance Policy")
        self.assertEqual(chunk.domain, "Academic Affairs")
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(
            chunk.chunk_text,
            "Students must maintain a minimum of 75.0% attendance across all registered courses.",
        )
        self.assertEqual(chunk.section, "1. Minimum Attendance Threshold")
        self.assertEqual(chunk.token_count, 55)
        self.assertEqual(chunk.score, 0.2435)
        self.assertEqual(chunk.metadata, {"department": "Academic Affairs", "status": "Active"})
        self.assertEqual(chunk.retriever_type, "keyword")

        # Test to_dict helper
        d = chunk.to_dict()
        self.assertEqual(d["chunk_id"], "POL-ATTEND_CHUNK_001")
        self.assertEqual(d["score"], 0.2435)
        self.assertEqual(d["retriever_type"], "keyword")

    # -------------------------------------------------------------------------
    # 9. RPC failure
    # -------------------------------------------------------------------------
    def test_09_rpc_failure(self) -> None:
        """Verify a useful RuntimeError is raised when client.rpc fails."""
        self.mock_client.rpc.side_effect = RuntimeError("PostgREST connection error")
        with self.assertRaises(RuntimeError) as ctx:
            self.retriever.retrieve("test query")
        self.assertIn("PostgREST connection error", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 10. Malformed response
    # -------------------------------------------------------------------------
    def test_10_malformed_response_not_list(self) -> None:
        """RPC returns something other than a list. Verify clear RuntimeError."""
        self.mock_client.rpc.return_value = {"error": "unexpected object format"}
        with self.assertRaises(RuntimeError) as ctx:
            self.retriever.retrieve("test query")
        self.assertIn("expected list", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # 11. Missing required field
    # -------------------------------------------------------------------------
    def test_11_missing_required_field(self) -> None:
        """Verify malformed records with missing fields are rejected."""
        bad_row = dict(self.sample_row)
        del bad_row["keyword_score"]
        self.mock_client.rpc.return_value = [bad_row]

        with self.assertRaises(RuntimeError) as ctx:
            self.retriever.retrieve("test query")
        self.assertIn("missing required field", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # 12. Raw query preserved
    # -------------------------------------------------------------------------
    def test_12_raw_query_preserved(self) -> None:
        """Verify raw query string with quotes, operators, and special characters is preserved."""
        complex_query = '"minimum prescribed attendance threshold" OR attend -medical'
        self.retriever.retrieve(complex_query)
        self.mock_client.rpc.assert_called_once_with(
            "match_chunks_keyword",
            {
                "query_text": complex_query,
                "match_count": 20,
                "filter_domain": None,
            },
        )

    # -------------------------------------------------------------------------
    # 13. No match_threshold parameter
    # -------------------------------------------------------------------------
    def test_13_no_threshold_parameter(self) -> None:
        """Verify KeywordRetriever.retrieve does not accept match_threshold parameter."""
        sig = inspect.signature(self.retriever.retrieve)
        self.assertNotIn("match_threshold", sig.parameters)
        with self.assertRaises(TypeError):
            self.retriever.retrieve("test query", match_threshold=0.5)  # type: ignore

    # -------------------------------------------------------------------------
    # 14. Invalid filter_domain parameter
    # -------------------------------------------------------------------------
    def test_14_invalid_filter_domain(self) -> None:
        """Verify filter_domain must be string or None."""
        with self.assertRaises(ValueError):
            self.retriever.retrieve("test query", filter_domain=123)  # type: ignore


if __name__ == "__main__":
    unittest.main()
