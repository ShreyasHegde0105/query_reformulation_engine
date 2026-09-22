from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

logger = logging.getLogger("embedding.validator")


class ChunkIngestionValidator:
    """Rigorous pre-ingestion validation for chunks and generated embeddings."""

    @classmethod
    def load_and_validate_jsonl(
        cls,
        chunks_path: Path | str,
        expected_chunk_count: int = 696,
        expected_doc_count: int = 157,
        validated_doc_ids: Set[str] | None = None,
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Reads and validates Data/chunks/chunks.jsonl.
        
        Returns:
            Tuple of (valid_chunks, error_messages).
        """
        p = Path(chunks_path)
        if not p.exists():
            return [], [f"Chunks file does not exist at '{p}'"]

        chunks: List[Dict[str, Any]] = []
        errors: List[str] = []
        seen_chunk_ids: Set[str] = set()
        seen_doc_ids: Set[str] = set()

        with open(p, "r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"Line {line_no}: Malformed JSON record: {exc}")
                    continue

                # 1. Verify document_id
                doc_id = record.get("document_id")
                if not doc_id or not isinstance(doc_id, str) or not doc_id.strip():
                    errors.append(f"Line {line_no}: Missing or empty 'document_id'")
                else:
                    seen_doc_ids.add(doc_id.strip())

                # 2. Verify chunk_id
                chunk_id = record.get("chunk_id")
                if not chunk_id or not isinstance(chunk_id, str) or not chunk_id.strip():
                    errors.append(f"Line {line_no}: Missing or empty 'chunk_id'")
                else:
                    chunk_id = chunk_id.strip()
                    if chunk_id in seen_chunk_ids:
                        errors.append(f"Line {line_no}: Duplicate chunk_id '{chunk_id}' detected")
                    seen_chunk_ids.add(chunk_id)

                # 3. Verify chunk text is non-empty
                text = record.get("text")
                if not text or not isinstance(text, str) or not text.strip():
                    errors.append(
                        f"Line {line_no} (chunk_id={chunk_id}): Missing or empty chunk 'text'"
                    )

                chunks.append(record)

        # 4. Verify chunk count
        if expected_chunk_count > 0 and len(chunks) != expected_chunk_count:
            errors.append(
                f"Expected {expected_chunk_count} chunks, but parsed {len(chunks)} records"
            )

        # 5. Verify document correspondence with validated corpus
        if expected_doc_count > 0 and len(seen_doc_ids) != expected_doc_count:
            errors.append(
                f"Expected {expected_doc_count} unique document IDs across chunks, but found {len(seen_doc_ids)}"
            )

        if validated_doc_ids:
            unmatched_docs = seen_doc_ids - validated_doc_ids
            if unmatched_docs:
                errors.append(
                    f"Chunk documents not found in validated corpus: {sorted(list(unmatched_docs))[:5]}"
                )
            missing_docs = validated_doc_ids - seen_doc_ids
            if missing_docs:
                errors.append(
                    f"Validated corpus documents missing from chunks: {sorted(list(missing_docs))[:5]}"
                )

        return chunks, errors

    @classmethod
    def validate_embeddings(
        cls,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        expected_dim: int = 384,
    ) -> List[str]:
        """Validate embedding vectors prior to database insertion."""
        errors: List[str] = []

        if len(chunks) != len(embeddings):
            errors.append(
                f"Chunk count ({len(chunks)}) does not match embedding count ({len(embeddings)})"
            )
            return errors

        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = chunk.get("chunk_id", f"index_{idx}")

            if len(emb) != expected_dim:
                errors.append(
                    f"Chunk '{chunk_id}' embedding dimension is {len(emb)}, expected {expected_dim}"
                )

            # Check values
            norm_sq = 0.0
            has_invalid = False
            for v in emb:
                if v is None or not isinstance(v, (int, float)):
                    has_invalid = True
                    break
                if v != v or abs(v) > 1e10:  # NaN or Inf check
                    has_invalid = True
                    break
                norm_sq += v * v

            if has_invalid:
                errors.append(f"Chunk '{chunk_id}' embedding contains NaN, None, or infinite values")
            else:
                norm = norm_sq ** 0.5
                if not (0.98 <= norm <= 1.02):
                    errors.append(f"Chunk '{chunk_id}' embedding is unnormalized (norm: {norm:.4f})")

        return errors
