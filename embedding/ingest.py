from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from embedding.client import SupabaseRestClient
from embedding.config import IngestionConfig
from embedding.encoder import EmbeddingEncoder
from embedding.mapper import DocumentMapper
from embedding.validator import ChunkIngestionValidator

logger = logging.getLogger("embedding.ingest")


def expected_chunk_count(chunks_path: Path) -> int:
    """Chunk count recorded by the chunker in ``chunking_report.json`` next to the chunks
    file, so a truncated or stale ``chunks.jsonl`` is caught. 0 (no check) if there is no report."""
    report_path = Path(chunks_path).parent / "chunking_report.json"
    try:
        with open(report_path, encoding="utf-8") as f:
            return int(json.load(f)["statistics"]["total_chunks"])
    except (OSError, ValueError, KeyError, TypeError):
        logger.warning("No chunk count in '%s'; skipping the chunk count check.", report_path)
        return 0


class IngestionPipeline:
    """Production ingestion pipeline for BGE embedding generation and Supabase storage."""

    def __init__(self, config: Optional[IngestionConfig] = None):
        self.config = config or IngestionConfig.from_env()
        self.encoder = EmbeddingEncoder(
            model_name=self.config.embedding_model_name,
            expected_dimension=self.config.embedding_dimension,
            batch_size=self.config.embedding_batch_size,
            local_files_only=self.config.local_files_only,
        )

    def run(self) -> Dict[str, Any]:
        """Execute the complete ingestion pipeline."""
        start_time = time.time()
        logger.info("==================================================")
        logger.info("  STARTING BGE EMBEDDING & SUPABASE INGESTION")
        logger.info("==================================================")
        logger.info("Model: %s", self.config.embedding_model_name)
        logger.info("Embedding Dimension: %d", self.config.embedding_dimension)
        logger.info("Chunks Input: %s", self.config.input_chunks_path)
        logger.info("Validated Docs: %s", self.config.validated_docs_path)
        logger.info("Supabase URL: %s", self.config.supabase_url or "(Not Set / Dry-Run Mode)")
        logger.info("Dry-Run Mode: %s", self.config.dry_run)
        logger.info("--------------------------------------------------")

        # 1. Load validated documents for cross-verification
        validated_docs_map = DocumentMapper.load_validated_documents(self.config.validated_docs_path)
        validated_doc_ids = set(validated_docs_map.keys())
        expected_docs = len(validated_doc_ids)
        logger.info("Loaded %d validated parent documents from '%s'.", len(validated_docs_map), self.config.validated_docs_path)

        # 2. Read and validate chunks.jsonl
        logger.info("Reading chunks...")
        chunks, val_errors = ChunkIngestionValidator.load_and_validate_jsonl(
            chunks_path=self.config.input_chunks_path,
            expected_chunk_count=expected_chunk_count(self.config.input_chunks_path),
            expected_doc_count=expected_docs,
            validated_doc_ids=validated_doc_ids if validated_doc_ids else None,
        )

        if val_errors:
            logger.error("Chunk validation failed with %d error(s):", len(val_errors))
            for err in val_errors[:10]:
                logger.error("  - %s", err)
            if len(val_errors) > 10:
                logger.error("  ... and %d more errors.", len(val_errors) - 10)
            raise ValueError(f"Pre-ingestion validation failed: {val_errors[0]}")

        logger.info("Validated %d chunks across %d unique documents.", len(chunks), expected_docs)

        # 3. Load embedding model and generate embeddings
        logger.info("Loading embedding model...")
        texts = [DocumentMapper.passage_text(chunk, self.config.contextual_passages) for chunk in chunks]
        logger.info("Contextual passages (title + section header): %s", self.config.contextual_passages)

        logger.info("Generating embeddings...")
        t0_emb = time.time()
        embeddings = self.encoder.encode_passages(
            texts=texts,
            batch_size=self.config.embedding_batch_size,
            show_progress=True,
        )
        emb_elapsed = time.time() - t0_emb
        logger.info("Generated %d embeddings in %.2f seconds.", len(embeddings), emb_elapsed)

        # 4. Rigorous post-generation embedding validation
        logger.info("Validating embedding shapes, values, and normalization...")
        emb_errors = ChunkIngestionValidator.validate_embeddings(
            chunks=chunks,
            embeddings=embeddings,
            expected_dim=self.config.embedding_dimension,
        )
        if emb_errors:
            logger.error("Embedding validation failed with %d error(s):", len(emb_errors))
            for err in emb_errors[:10]:
                logger.error("  - %s", err)
            raise ValueError(f"Embedding validation failed: {emb_errors[0]}")
        logger.info("Embeddings verified: all 384 dimensions, non-null, unit-normalized.")

        # 5. Connect to Supabase or execute dry run
        if self.config.dry_run or not self.config.supabase_url or not self.config.supabase_key:
            logger.info("Dry-run mode or Supabase credentials not set.")
            logger.info("Skipping network insertion. Mapping and data verification completed.")
            doc_records = DocumentMapper.build_document_records(chunks, validated_docs_map)
            chunk_records = DocumentMapper.build_chunk_records(chunks, embeddings)
            return {
                "status": "DRY_RUN_COMPLETED",
                "model": self.config.embedding_model_name,
                "dimension": self.config.embedding_dimension,
                "documents_count": len(doc_records),
                "chunks_count": len(chunk_records),
                "embeddings_count": len(embeddings),
                "elapsed_seconds": round(time.time() - start_time, 2),
            }

        client = SupabaseRestClient(
            supabase_url=self.config.supabase_url,
            api_key=self.config.supabase_key,
            timeout=self.config.timeout,
        )

        try:
            # 6. Introspect table columns to adapt dynamically
            logger.info("Introspecting existing Supabase schema...")
            table_cols = client.introspect_columns()
            doc_allowed_cols = table_cols.get("documents") or None
            chunk_allowed_cols = table_cols.get("document_chunks") or None

            # 7. Map document records and upsert
            logger.info("Upserting documents...")
            doc_records = DocumentMapper.build_document_records(
                chunks=chunks,
                validated_docs_map=validated_docs_map,
                allowed_columns=doc_allowed_cols,
            )

            # Batch upsert documents
            db_bs = self.config.db_batch_size
            total_docs = len(doc_records)
            for i in range(0, total_docs, db_bs):
                batch = doc_records[i : i + db_bs]
                client.upsert_batch("documents", batch, on_conflict="document_id")
                logger.info("Upserted documents batch %d-%d of %d.", i + 1, min(i + db_bs, total_docs), total_docs)

            # 8. Map chunk records and upsert
            logger.info("Upserting chunks...")
            chunk_records = DocumentMapper.build_chunk_records(
                chunks=chunks,
                embeddings=embeddings,
                allowed_columns=chunk_allowed_cols,
            )

            total_chunks = len(chunk_records)
            for i in range(0, total_chunks, db_bs):
                batch = chunk_records[i : i + db_bs]
                client.upsert_batch("document_chunks", batch, on_conflict="chunk_id")
                logger.info("Upserted chunks batch %d-%d of %d.", i + 1, min(i + db_bs, total_chunks), total_chunks)

            # 9. Verify Supabase database integrity
            logger.info("Verifying Supabase...")
            verification = client.verify_database_integrity(
                expected_docs=expected_docs,
                expected_chunks=len(chunks),
                expected_dim=self.config.embedding_dimension,
            )

            logger.info("Database verification status: %s", "PASSED" if verification["is_valid"] else "FAILED")
            logger.info("Documents in DB: %d (Expected: %d)", verification["document_count"], expected_docs)
            logger.info("Chunks in DB:    %d (Expected: %d)", verification["chunk_count"], len(chunks))

            logger.info("--- Sample Retrieved Chunks ---")
            for sample in verification["samples"]:
                logger.info(
                    "Doc: %s | Chunk: %s | Sec: %s | Tokens: %s | Dim: %d",
                    sample.get("document_id"),
                    sample.get("chunk_id"),
                    sample.get("section") or "N/A",
                    sample.get("token_count"),
                    sample.get("embedding_dim"),
                )

            if not verification["is_valid"]:
                raise RuntimeError(
                    f"Supabase verification mismatch: docs={verification['document_count']}/{expected_docs}, "
                    f"chunks={verification['chunk_count']}/{len(chunks)}"
                )

            elapsed = round(time.time() - start_time, 2)
            logger.info("Ingestion completed successfully in %.2f seconds.", elapsed)

            return {
                "status": "COMPLETED",
                "model": self.config.embedding_model_name,
                "dimension": self.config.embedding_dimension,
                "documents_count": verification["document_count"],
                "chunks_count": verification["chunk_count"],
                "embeddings_count": len(embeddings),
                "verification": verification,
                "elapsed_seconds": elapsed,
            }

        finally:
            client.close()
