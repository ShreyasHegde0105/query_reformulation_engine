from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from embedding.client import SupabaseRestClient
from embedding.encoder import EmbeddingEncoder
from retrieval.models import RetrievedChunk
from retrieval.vector_retriever import VectorRetriever


class TestVectorRetriever(unittest.TestCase):
    """Unit tests for VectorRetriever using mocked dependencies."""

    def setUp(self) -> None:
        self.mock_encoder = MagicMock(spec=EmbeddingEncoder)
        self.dummy_embedding = [0.05] * 384
        self.mock_encoder.embed_query.return_value = self.dummy_embedding

        self.mock_client = MagicMock(spec=SupabaseRestClient)
        self.sample_row = {
            "chunk_id": "POL-ATTEND_CHUNK_001",
            "document_id": "POL-ATTEND",
            "title": "Comprehensive Academic Attendance Policy",
            "domain": "Academic Affairs",
            "chunk_index": 0,
            "chunk_text": "Students must maintain a minimum of 75.0% attendance.",
            "section": "1. Minimum Attendance Threshold",
            "token_count": 55,
            "similarity": 0.885,
            "metadata": {"department": "Academic Affairs", "status": "Active"},
        }
        self.mock_client.rpc.return_value = [self.sample_row]
        self.retriever = VectorRetriever(encoder=self.mock_encoder, client=self.mock_client)

    # -------------------------------------------------------------------------
    # 1. Successful retrieval
    # -------------------------------------------------------------------------
    def test_01_successful_retrieval(self) -> None:
        """Verify that a valid query returns RetrievedChunk objects."""
        results = self.retriever.retrieve("attendance policy")
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], RetrievedChunk)
        self.assertEqual(results[0].chunk_id, "POL-ATTEND_CHUNK_001")
        self.assertEqual(results[0].score, 0.885)
        self.assertEqual(results[0].retriever_type, "vector")

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
        """Test 0, negative values, values > 100, and non-ints."""
        # 0
        with self.assertRaises(ValueError):
            self.retriever.retrieve("test", match_count=0)
        # Negative
        with self.assertRaises(ValueError):
            self.retriever.retrieve("test", match_count=-5)
        # Above 100
        with self.assertRaises(ValueError):
            self.retriever.retrieve("test", match_count=101)
        # Boolean trap (True is an int subclass in Python)
        with self.assertRaises(ValueError):
            self.retriever.retrieve("test", match_count=True)

    # -------------------------------------------------------------------------
    # 5. Invalid threshold
    # -------------------------------------------------------------------------
    def test_05_invalid_threshold(self) -> None:
        """Test below 0 and above 1."""
        with self.assertRaises(ValueError):
            self.retriever.retrieve("test", match_threshold=-0.01)
        with self.assertRaises(ValueError):
            self.retriever.retrieve("test", match_threshold=1.01)
        # Boolean trap
        with self.assertRaises(ValueError):
            self.retriever.retrieve("test", match_threshold=True)

    # -------------------------------------------------------------------------
    # 6. RPC payload
    # -------------------------------------------------------------------------
    def test_06_rpc_payload(self) -> None:
        """Verify match_chunks_vector is called with exactly expected payload."""
        self.retriever.retrieve("hostel fee payment", match_count=15, match_threshold=0.45)
        self.mock_client.rpc.assert_called_once_with(
            "match_chunks_vector",
            {
                "query_embedding": self.dummy_embedding,
                "match_threshold": 0.45,
                "match_count": 15,
                "filter_domain": None,
            },
        )

    # -------------------------------------------------------------------------
    # 7. Domain filter
    # -------------------------------------------------------------------------
    def test_07_domain_filter_passed(self) -> None:
        """Verify filter_domain is passed correctly in payload."""
        self.retriever.retrieve("examination rules", filter_domain="Examination Cell")
        self.mock_client.rpc.assert_called_once_with(
            "match_chunks_vector",
            {
                "query_embedding": self.dummy_embedding,
                "match_threshold": 0.0,
                "match_count": 20,
                "filter_domain": "Examination Cell",
            },
        )

    # -------------------------------------------------------------------------
    # 8. Zero results
    # -------------------------------------------------------------------------
    def test_08_zero_results(self) -> None:
        """RPC returns []. VectorRetriever should return []."""
        self.mock_client.rpc.return_value = []
        results = self.retriever.retrieve("unmatched query term")
        self.assertEqual(results, [])

    # -------------------------------------------------------------------------
    # 9. Result mapping
    # -------------------------------------------------------------------------
    def test_09_result_mapping(self) -> None:
        """Verify all returned database fields map correctly to RetrievedChunk."""
        results = self.retriever.retrieve("attendance policy")
        chunk = results[0]
        self.assertEqual(chunk.chunk_id, "POL-ATTEND_CHUNK_001")
        self.assertEqual(chunk.document_id, "POL-ATTEND")
        self.assertEqual(chunk.title, "Comprehensive Academic Attendance Policy")
        self.assertEqual(chunk.domain, "Academic Affairs")
        self.assertEqual(chunk.chunk_index, 0)
        self.assertEqual(chunk.chunk_text, "Students must maintain a minimum of 75.0% attendance.")
        self.assertEqual(chunk.section, "1. Minimum Attendance Threshold")
        self.assertEqual(chunk.token_count, 55)
        self.assertEqual(chunk.score, 0.885)
        self.assertEqual(chunk.metadata, {"department": "Academic Affairs", "status": "Active"})
        self.assertEqual(chunk.retriever_type, "vector")

        # Test to_dict helper
        d = chunk.to_dict()
        self.assertEqual(d["chunk_id"], "POL-ATTEND_CHUNK_001")
        self.assertEqual(d["score"], 0.885)
        self.assertEqual(d["retriever_type"], "vector")

    # -------------------------------------------------------------------------
    # 10. RPC failure
    # -------------------------------------------------------------------------
    def test_10_rpc_failure(self) -> None:
        """Verify a useful exception is raised when client.rpc fails."""
        self.mock_client.rpc.side_effect = RuntimeError("PostgREST connection timeout")
        with self.assertRaises(RuntimeError) as ctx:
            self.retriever.retrieve("test query")
        self.assertIn("PostgREST connection timeout", str(ctx.exception))

    # -------------------------------------------------------------------------
    # 11. Malformed response
    # -------------------------------------------------------------------------
    def test_11_malformed_response_not_list(self) -> None:
        """RPC returns something other than a list. Verify clear RuntimeError."""
        self.mock_client.rpc.return_value = {"error": "unexpected object format"}
        with self.assertRaises(RuntimeError) as ctx:
            self.retriever.retrieve("test query")
        self.assertIn("expected list", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # 12. Missing required field
    # -------------------------------------------------------------------------
    def test_12_missing_required_field(self) -> None:
        """Verify malformed records with missing fields are rejected."""
        bad_row = dict(self.sample_row)
        del bad_row["similarity"]
        self.mock_client.rpc.return_value = [bad_row]

        with self.assertRaises(RuntimeError) as ctx:
            self.retriever.retrieve("test query")
        self.assertIn("missing required field", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # 13. Encoder invocation
    # -------------------------------------------------------------------------
    def test_13_encoder_invocation(self) -> None:
        """Verify encoder.embed_query is called exactly once with raw query without prefix."""
        raw_query = "how to apply for condonation?"
        self.retriever.retrieve(raw_query)

        self.mock_encoder.embed_query.assert_called_once_with(raw_query)
        # Ensure raw query was NOT modified with instruction prefix before calling embed_query
        called_arg = self.mock_encoder.embed_query.call_args[0][0]
        self.assertEqual(called_arg, raw_query)
        self.assertFalse(called_arg.startswith("Represent this sentence"))


if __name__ == "__main__":
    unittest.main()
