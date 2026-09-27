from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from embedding.config import IngestionConfig
from embedding.client import KNOWN_DOC_COLS, KNOWN_CHUNK_COLS, SupabaseRestClient
from embedding.encoder import EmbeddingEncoder, BGE_QUERY_INSTRUCTION
from embedding.mapper import DocumentMapper
from embedding.validator import ChunkIngestionValidator
from embedding.ingest import IngestionPipeline


class TestEmbeddingAndIngestion(unittest.TestCase):
    """Focused unit and integration tests for the embedding and ingestion module."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # -------------------------------------------------------------
    # Configuration Tests
    # -------------------------------------------------------------
    def test_config_defaults(self) -> None:
        config = IngestionConfig.from_env()
        self.assertEqual(config.embedding_model_name, "BAAI/bge-small-en-v1.5")
        self.assertEqual(config.embedding_dimension, 384)
        self.assertEqual(config.embedding_batch_size, 32)
        self.assertEqual(config.db_batch_size, 50)
        self.assertEqual(config.input_chunks_path, Path("Data/chunks/chunks.jsonl"))

    def test_config_overrides(self) -> None:
        config = IngestionConfig.from_env(
            supabase_url="https://test.supabase.co",
            supabase_key="secret-key",
            embedding_batch_size=16,
            dry_run=True,
        )
        self.assertEqual(config.supabase_url, "https://test.supabase.co")
        self.assertEqual(config.supabase_key, "secret-key")
        self.assertEqual(config.embedding_batch_size, 16)
        self.assertTrue(config.dry_run)

    # -------------------------------------------------------------
    # Validator Tests
    # -------------------------------------------------------------
    def test_validator_detects_malformed_json(self) -> None:
        bad_file = self.temp_path / "bad.jsonl"
        with open(bad_file, "w", encoding="utf-8") as f:
            f.write('{"chunk_id": "C1", "document_id": "D1", "text": "OK"}\n')
            f.write('INVALID JSON LINE\n')

        chunks, errors = ChunkIngestionValidator.load_and_validate_jsonl(
            bad_file, expected_chunk_count=2, expected_doc_count=1
        )
        self.assertEqual(len(chunks), 1)
        self.assertTrue(any("Malformed JSON record" in e for e in errors))

    def test_validator_detects_missing_fields(self) -> None:
        bad_file = self.temp_path / "missing_fields.jsonl"
        with open(bad_file, "w", encoding="utf-8") as f:
            # Missing document_id
            f.write(json.dumps({"chunk_id": "C1", "text": "Some text"}) + "\n")
            # Missing chunk_id
            f.write(json.dumps({"document_id": "D1", "text": "Some text"}) + "\n")
            # Empty text
            f.write(json.dumps({"chunk_id": "C3", "document_id": "D1", "text": "   "}) + "\n")

        chunks, errors = ChunkIngestionValidator.load_and_validate_jsonl(
            bad_file, expected_chunk_count=3, expected_doc_count=1
        )
        self.assertTrue(any("Missing or empty 'document_id'" in e for e in errors))
        self.assertTrue(any("Missing or empty 'chunk_id'" in e for e in errors))
        self.assertTrue(any("Missing or empty chunk 'text'" in e for e in errors))

    def test_validator_detects_duplicate_chunk_ids(self) -> None:
        dup_file = self.temp_path / "dup.jsonl"
        with open(dup_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"chunk_id": "C1", "document_id": "D1", "text": "Text 1"}) + "\n")
            f.write(json.dumps({"chunk_id": "C1", "document_id": "D1", "text": "Text 2"}) + "\n")

        chunks, errors = ChunkIngestionValidator.load_and_validate_jsonl(
            dup_file, expected_chunk_count=2, expected_doc_count=1
        )
        self.assertTrue(any("Duplicate chunk_id 'C1'" in e for e in errors))

    def test_validator_detects_invalid_embeddings(self) -> None:
        chunks = [{"chunk_id": "C1"}, {"chunk_id": "C2"}]
        # Wrong dimension
        embs_wrong_dim = [[0.1] * 383, [0.1] * 384]
        errs = ChunkIngestionValidator.validate_embeddings(chunks, embs_wrong_dim, expected_dim=384)
        self.assertTrue(any("embedding dimension is 383" in e for e in errs))

        # NaN / Inf
        embs_nan = [[0.1] * 383 + [float("nan")], [0.1] * 384]
        errs_nan = ChunkIngestionValidator.validate_embeddings(chunks, embs_nan, expected_dim=384)
        self.assertTrue(any("contains NaN, None, or infinite" in e for e in errs_nan))

        # Unnormalized (norm >> 1.0)
        embs_unnorm = [[1.0] * 384, [0.0] * 384]
        errs_unnorm = ChunkIngestionValidator.validate_embeddings(chunks, embs_unnorm, expected_dim=384)
        self.assertTrue(any("unnormalized" in e for e in errs_unnorm))

    # -------------------------------------------------------------
    # Encoder Tests
    # -------------------------------------------------------------
    def test_encoder_passage_generation(self) -> None:
        encoder = EmbeddingEncoder(local_files_only=True)
        texts = ["Demo policy text for mandatory attendance requirement."]
        embs = encoder.encode_passages(texts, show_progress=False)
        self.assertEqual(len(embs), 1)
        self.assertEqual(len(embs[0]), 384)
        norm = sum(x * x for x in embs[0]) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=2)

    def test_encoder_query_generation(self) -> None:
        encoder = EmbeddingEncoder(local_files_only=True)
        query = "what is the minimum attendance required?"
        emb = encoder.embed_query(query)
        self.assertEqual(len(emb), 384)
        norm = sum(x * x for x in emb) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=2)

    def test_encoder_query_cache(self) -> None:
        from unittest.mock import MagicMock
        import numpy as np

        encoder = EmbeddingEncoder(query_cache_size=2)
        model = MagicMock()
        model.encode.side_effect = lambda text, normalize_embeddings=True: np.eye(384)[len(text) % 384]
        encoder._model = model

        first = encoder.embed_query("hostel fees")
        first.append(99.0)  # callers mutating the result must not corrupt the cache
        self.assertEqual(len(encoder.embed_query("  hostel fees  ")), 384)
        self.assertEqual(model.encode.call_count, 1)

        encoder.embed_query("library")
        encoder.embed_query("exam")          # evicts "hostel fees" (least recently used)
        encoder.embed_query("hostel fees")
        self.assertEqual(model.encode.call_count, 4)

        uncached = EmbeddingEncoder(query_cache_size=0)
        uncached._model = model
        uncached.embed_query("x"); uncached.embed_query("x")
        self.assertEqual(model.encode.call_count, 6)

    # -------------------------------------------------------------
    # Mapper Tests (Confirmed Supabase Schema)
    # -------------------------------------------------------------
    def test_mapper_document_records(self) -> None:
        chunks = [
            {
                "document_id": "DOC-1",
                "title": "Title 1",
                "department": "Engineering",
                "department_id": "DEPT-ENG",
                "document_type": "Policy",
                "version": "1.0",
                "effective_date": "2026-01-01",
                "status": "Active",
                "audience": ["Students"],
                "keywords": ["test"],
                "retrieval_signals": ["sig"],
                "systems": ["ERP"],
                "roles": ["Dean"],
                "related_documents": [],
            },
            {
                "document_id": "DOC-1",
                "title": "Title 1",
                "department": "Engineering",
                "department_id": "DEPT-ENG",
                "document_type": "Policy",
                "version": "1.0",
                "effective_date": "2026-01-01",
                "status": "Active",
                "audience": ["Students"],
                "keywords": ["test"],
                "retrieval_signals": ["sig"],
                "systems": ["ERP"],
                "roles": ["Dean"],
                "related_documents": [],
            },
        ]
        doc_records = DocumentMapper.build_document_records(chunks)
        self.assertEqual(len(doc_records), 1)
        doc = doc_records[0]
        self.assertEqual(doc["document_id"], "DOC-1")
        self.assertEqual(doc["title"], "Title 1")
        self.assertEqual(doc["domain"], "Engineering")  # department -> domain
        self.assertNotIn("department", doc)  # department not top-level
        self.assertNotIn("department_id", doc)  # nested in metadata
        self.assertIsNone(doc["storage_path"])
        self.assertEqual(set(doc.keys()), KNOWN_DOC_COLS)
        self.assertEqual(doc["metadata"]["department"], "Engineering")
        self.assertEqual(doc["metadata"]["document_type"], "Policy")
        self.assertEqual(doc["metadata"]["version"], "1.0")
        self.assertEqual(doc["metadata"]["status"], "Active")
        self.assertIn("audience", doc["metadata"])
        self.assertIn("keywords", doc["metadata"])

    def test_mapper_chunk_records(self) -> None:
        chunks = [
            {
                "chunk_id": "DOC-1_CHUNK_001",
                "document_id": "DOC-1",
                "text": "Chunk text body",
                "section": "1. SCOPE",
                "section_number": "1",
                "subsection": None,
                "chunk_index": 0,
                "total_chunks": 1,
                "token_count": 10,
                "character_count": 15,
                "start_char": 0,
                "end_char": 15,
            }
        ]
        embs = [[0.05] * 384]
        records = DocumentMapper.build_chunk_records(chunks, embs)
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r["chunk_id"], "DOC-1_CHUNK_001")
        self.assertEqual(r["document_id"], "DOC-1")
        self.assertEqual(r["chunk_text"], "Chunk text body")  # text -> chunk_text
        self.assertNotIn("text", r)  # text not top-level
        self.assertNotIn("total_chunks", r)  # nested in metadata
        self.assertEqual(r["embedding"], embs[0])
        self.assertEqual(r["section"], "1. SCOPE")
        self.assertEqual(set(r.keys()), KNOWN_CHUNK_COLS)
        self.assertEqual(r["metadata"]["total_chunks"], 1)
        self.assertEqual(r["metadata"]["character_count"], 15)

    def test_mapper_schema_adaptation(self) -> None:
        chunks = [
            {
                "chunk_id": "C1",
                "document_id": "D1",
                "text": "Hello",
                "section": "Sec",
                "chunk_index": 0,
                "total_chunks": 1,
                "token_count": 2,
                "character_count": 5,
            }
        ]
        embs = [[0.1] * 384]
        # Only allow chunk_id, document_id, chunk_text, embedding
        allowed = {"chunk_id", "document_id", "chunk_text", "embedding"}
        records = DocumentMapper.build_chunk_records(chunks, embs, allowed_columns=allowed)
        self.assertEqual(set(records[0].keys()), allowed)
        self.assertEqual(records[0]["chunk_text"], "Hello")

    # -------------------------------------------------------------
    # Client Fallback Columns Test
    # -------------------------------------------------------------
    def test_client_fallback_columns(self) -> None:
        client = SupabaseRestClient(supabase_url="https://dummy.supabase.co", api_key="dummy_key")
        with patch.object(client.client, "get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 401
            mock_get.return_value = mock_resp

            cols = client.introspect_columns()
            self.assertEqual(cols["documents"], KNOWN_DOC_COLS)
            self.assertEqual(cols["document_chunks"], KNOWN_CHUNK_COLS)

    # -------------------------------------------------------------
    # Pipeline Dry-Run Test
    # -------------------------------------------------------------
    def test_pipeline_dry_run(self) -> None:
        cfg = IngestionConfig.from_env(dry_run=True, embedding_batch_size=64)
        pipeline = IngestionPipeline(config=cfg)
        # Mock encoder.encode_passages to keep unit test fast (norm = 1.0)
        unit_val = 1.0 / (384 ** 0.5)
        with patch.object(pipeline.encoder, "encode_passages") as mock_encode:
            mock_encode.return_value = [[unit_val] * 384] * 696
            res = pipeline.run()

        self.assertEqual(res["status"], "DRY_RUN_COMPLETED")
        self.assertEqual(res["documents_count"], 157)
        self.assertEqual(res["chunks_count"], 696)
        self.assertEqual(res["embeddings_count"], 696)
        self.assertEqual(res["dimension"], 384)


if __name__ == "__main__":
    unittest.main()
