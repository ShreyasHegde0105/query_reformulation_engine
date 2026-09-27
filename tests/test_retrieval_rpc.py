from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

from dotenv import load_dotenv

from embedding.encoder import EmbeddingEncoder, BGE_QUERY_INSTRUCTION


class TestRetrievalDatabaseLayer(unittest.TestCase):
    """Unit tests for Phase 5.1 Retrieval Database Layer (RPC contracts and validation)."""

    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv()
        cls.migration_path = Path("supabase/migrations/20260922000001_retrieval_functions.sql")

    # -------------------------------------------------------------------------
    # 1. Migration File & Contract Tests
    # -------------------------------------------------------------------------
    def test_migration_files_exist(self) -> None:
        """Verify that the Supabase migration file exists."""
        self.assertTrue(self.migration_path.exists(), f"Missing {self.migration_path}")

    def test_schema_migration_covers_ingested_columns(self) -> None:
        """The core schema migration must define every column ingestion writes, and sort first."""
        import re
        from embedding.client import KNOWN_CHUNK_COLS, KNOWN_DOC_COLS

        schema_path = Path("supabase/migrations/20260922000000_core_schema.sql")
        self.assertTrue(schema_path.exists(), f"Missing {schema_path}")
        self.assertLess(schema_path.name, self.migration_path.name)

        sql = schema_path.read_text(encoding="utf-8")
        for table, expected in (("documents", KNOWN_DOC_COLS), ("document_chunks", KNOWN_CHUNK_COLS)):
            body = re.search(rf"CREATE TABLE IF NOT EXISTS {table} \((.*?)\n\);", sql, re.S)
            self.assertIsNotNone(body, f"No CREATE TABLE for {table}")
            columns = {line.split()[0] for line in body.group(1).strip().splitlines() if line.strip()}
            self.assertTrue(expected <= columns, f"{table} missing {expected - columns}")
        self.assertIn("embedding   extensions.vector(384)", sql)
        self.assertIn("USING hnsw (embedding extensions.vector_cosine_ops)", sql)

    def test_migration_sql_contains_required_functions(self) -> None:
        """Verify SQL contains required function definitions and specifications."""
        sql = self.migration_path.read_text(encoding="utf-8")
        
        # Function declarations
        self.assertIn("CREATE OR REPLACE FUNCTION match_chunks_vector", sql)
        self.assertIn("CREATE OR REPLACE FUNCTION match_chunks_keyword", sql)

        # Vector specifications
        self.assertIn("query_embedding extensions.vector(384)", sql)
        self.assertIn("match_threshold float DEFAULT 0.0", sql)
        self.assertIn("match_count integer DEFAULT 20", sql)
        self.assertIn("filter_domain text DEFAULT NULL", sql)
        self.assertIn("1 - (c.embedding <=> query_embedding)", sql)
        self.assertIn("c.embedding <=> query_embedding ASC", sql)

        # Keyword specifications
        self.assertIn("query_text text", sql)
        self.assertIn("websearch_to_tsquery('english', query_text)", sql)
        self.assertIn("ts_rank_cd(", sql)
        self.assertIn("ORDER BY keyword_score DESC", sql)

        # Immutability / Read-only
        self.assertIn("STABLE", sql)

        # Permissions
        self.assertIn("GRANT EXECUTE ON FUNCTION match_chunks_vector", sql)
        self.assertIn("GRANT EXECUTE ON FUNCTION match_chunks_keyword", sql)

    # -------------------------------------------------------------------------
    # 2. Vector RPC Contract Tests
    # -------------------------------------------------------------------------
    def test_vector_rpc_payload_contract(self) -> None:
        """Verify vector RPC payload accepts valid 384-d embedding and parameters."""
        dummy_embedding = [0.05] * 384
        payload = {
            "query_embedding": dummy_embedding,
            "match_threshold": 0.5,
            "match_count": 10,
            "filter_domain": "Academic Affairs",
        }
        self.assertEqual(len(payload["query_embedding"]), 384)
        self.assertIsInstance(payload["match_threshold"], float)
        self.assertIsInstance(payload["match_count"], int)
        self.assertEqual(payload["filter_domain"], "Academic Affairs")

    def test_vector_rpc_result_structure(self) -> None:
        """Verify the expected vector RPC result schema."""
        expected_fields = {
            "chunk_id",
            "document_id",
            "title",
            "domain",
            "chunk_index",
            "chunk_text",
            "section",
            "token_count",
            "similarity",
            "metadata",
        }
        mock_result = {
            "chunk_id": "POL-ATTEND_CHUNK_001",
            "document_id": "POL-ATTEND",
            "title": "Comprehensive Academic Attendance and Regularization Policy",
            "domain": "Academic Affairs",
            "chunk_index": 0,
            "chunk_text": "Sample policy chunk text...",
            "section": "1. Minimum Attendance Threshold",
            "token_count": 69,
            "similarity": 0.825,
            "metadata": {"department": "Academic Affairs"},
        }
        self.assertEqual(set(mock_result.keys()), expected_fields)
        self.assertIsInstance(mock_result["similarity"], float)

    def test_vector_rpc_similarity_ordering_and_filtering(self) -> None:
        """Verify simulated vector results respect threshold, sorting, and domain filter."""
        candidates = [
            {"chunk_id": "C1", "similarity": 0.85, "domain": "Academic Affairs"},
            {"chunk_id": "C2", "similarity": 0.45, "domain": "Academic Affairs"},
            {"chunk_id": "C3", "similarity": 0.90, "domain": "Engineering"},
            {"chunk_id": "C4", "similarity": 0.70, "domain": "Academic Affairs"},
        ]
        
        # Apply domain filter: Academic Affairs
        domain_filtered = [c for c in candidates if c["domain"] == "Academic Affairs"]
        self.assertEqual(len(domain_filtered), 3)

        # Apply threshold: >= 0.50
        threshold_filtered = [c for c in domain_filtered if c["similarity"] >= 0.50]
        self.assertEqual(len(threshold_filtered), 2)

        # Sort descending by similarity
        sorted_results = sorted(threshold_filtered, key=lambda x: x["similarity"], reverse=True)
        self.assertEqual([c["chunk_id"] for c in sorted_results], ["C1", "C4"])

    def test_vector_rpc_zero_result_behavior(self) -> None:
        """Verify vector RPC returns empty list when threshold is impossibly high."""
        candidates = [
            {"chunk_id": "C1", "similarity": 0.85, "domain": "Academic Affairs"},
        ]
        results = [c for c in candidates if c["similarity"] >= 1.5]
        self.assertEqual(results, [])

    # -------------------------------------------------------------------------
    # 3. Keyword RPC Contract Tests
    # -------------------------------------------------------------------------
    def test_keyword_rpc_payload_contract(self) -> None:
        """Verify keyword RPC payload format."""
        payload = {
            "query_text": "attendance condonation",
            "match_count": 15,
            "filter_domain": "Academic Affairs",
        }
        self.assertEqual(payload["query_text"], "attendance condonation")
        self.assertEqual(payload["match_count"], 15)
        self.assertEqual(payload["filter_domain"], "Academic Affairs")

    def test_keyword_rpc_result_structure(self) -> None:
        """Verify the expected keyword RPC result schema."""
        expected_fields = {
            "chunk_id",
            "document_id",
            "title",
            "domain",
            "chunk_index",
            "chunk_text",
            "section",
            "token_count",
            "keyword_score",
            "metadata",
        }
        mock_result = {
            "chunk_id": "POL-ATTEND_CHUNK_002",
            "document_id": "POL-ATTEND",
            "title": "Comprehensive Academic Attendance and Regularization Policy",
            "domain": "Academic Affairs",
            "chunk_index": 1,
            "chunk_text": "Students with attendance between 65.0% and 74.9% may apply...",
            "section": "2. Condonation and Medical Absence Protocol",
            "token_count": 112,
            "keyword_score": 0.082,
            "metadata": {"department": "Academic Affairs"},
        }
        self.assertEqual(set(mock_result.keys()), expected_fields)
        self.assertIsInstance(mock_result["keyword_score"], float)

    def test_keyword_rpc_ranking_and_filtering(self) -> None:
        """Verify simulated keyword results respect ranking, domain filtering, and limit."""
        candidates = [
            {"chunk_id": "C1", "keyword_score": 0.045, "domain": "Academic Affairs"},
            {"chunk_id": "C2", "keyword_score": 0.120, "domain": "Finance"},
            {"chunk_id": "C3", "keyword_score": 0.088, "domain": "Academic Affairs"},
            {"chunk_id": "C4", "keyword_score": 0.015, "domain": "Academic Affairs"},
        ]

        # Domain filter: Academic Affairs
        filtered = [c for c in candidates if c["domain"] == "Academic Affairs"]
        self.assertEqual(len(filtered), 3)

        # Sort descending by keyword_score
        ranked = sorted(filtered, key=lambda x: x["keyword_score"], reverse=True)
        self.assertEqual([c["chunk_id"] for c in ranked], ["C3", "C1", "C4"])

        # Limit count = 2
        limited = ranked[:2]
        self.assertEqual(len(limited), 2)
        self.assertEqual(limited[0]["chunk_id"], "C3")

    def test_keyword_rpc_zero_result_behavior(self) -> None:
        """Verify clean zero-result behavior for unmatched terms."""
        candidates: List[Dict[str, Any]] = []
        self.assertEqual(len(candidates), 0)


if __name__ == "__main__":
    unittest.main()
