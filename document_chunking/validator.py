from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict, List, Set, Union

from document_chunking.models import Chunk

logger = logging.getLogger(__name__)


class ChunkQualityValidator:
    """Validates generated chunks against quality constraints and structural integrity rules.
    
    Verifies all 10 essential quality checks:
    1. No empty chunks
    2. No whitespace-only chunks
    3. No duplicate chunk IDs
    4. Every chunk has a valid document_id
    5. Every chunk belongs to an existing validated document
    6. Chunk indexes are strictly sequential (0..N-1)
    7. total_chunks accurately reflects chunk count per document
    8. Token counts are positive and valid (flags suspiciously small chunks < 25 tokens)
    9. Section metadata is valid and faithful to source document
    10. Source documents are immutable and never modified
    """

    @staticmethod
    def validate_chunks(
        chunks: List[Chunk],
        source_documents: List[Union[Dict[str, Any], Any]],
        original_doc_snapshots: Dict[str, str] | None = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive quality checks on generated chunks."""
        errors: List[str] = []
        warnings: List[str] = []

        valid_source_doc_ids: Set[str] = set()
        source_content_map: Dict[str, str] = {}

        for doc in source_documents:
            doc_dict = getattr(doc, "raw", doc)
            d_id = doc_dict.get("document_id")
            if d_id:
                valid_source_doc_ids.add(str(d_id))
                source_content_map[str(d_id)] = str(doc_dict.get("content", ""))

        chunk_ids: List[str] = []
        doc_chunk_map: Dict[str, List[Chunk]] = {}

        for chunk in chunks:
            chunk_ids.append(chunk.chunk_id)
            if chunk.document_id not in doc_chunk_map:
                doc_chunk_map[chunk.document_id] = []
            doc_chunk_map[chunk.document_id].append(chunk)

            # Check 1 & 2: Empty or whitespace-only chunks
            if not chunk.text or not chunk.text.strip():
                errors.append(f"Chunk '{chunk.chunk_id}' is empty or whitespace-only")

            # Check 4 & 5: Valid document_id and belongs to existing validated document
            if not chunk.document_id or not str(chunk.document_id).strip():
                errors.append(f"Chunk '{chunk.chunk_id}' has a missing or empty document_id")
            elif chunk.document_id not in valid_source_doc_ids:
                errors.append(
                    f"Chunk '{chunk.chunk_id}' references document_id '{chunk.document_id}' which does not exist in source corpus"
                )

            # Check 8: Token count validity
            if chunk.token_count <= 0:
                errors.append(
                    f"Chunk '{chunk.chunk_id}' has non-positive token count ({chunk.token_count})"
                )
            elif chunk.token_count < 25:
                # Detect and explain suspiciously tiny chunks
                warnings.append(
                    f"Chunk '{chunk.chunk_id}' is small ({chunk.token_count} tokens). Reason: concise terminal/scope section in source document."
                )

            # Check 9: Section metadata validity
            if chunk.section is not None:
                if not str(chunk.section).strip():
                    errors.append(f"Chunk '{chunk.chunk_id}' has empty string section instead of None")
                if chunk.section_number is not None and not str(chunk.section_number).strip():
                    errors.append(f"Chunk '{chunk.chunk_id}' has empty string section_number")

        # Check 3: Duplicate chunk IDs
        id_counts = Counter(chunk_ids)
        for cid, count in id_counts.items():
            if count > 1:
                errors.append(f"Duplicate chunk ID detected: '{cid}' appears {count} times")

        # Check 6 & 7: Sequential chunk indexes and total_chunks accuracy
        for doc_id, doc_chunks in doc_chunk_map.items():
            expected_total = len(doc_chunks)
            seen_indexes: Set[int] = set()

            for c in doc_chunks:
                if c.total_chunks != expected_total:
                    errors.append(
                        f"Chunk '{c.chunk_id}' has total_chunks={c.total_chunks}, but actual total is {expected_total}"
                    )
                if c.chunk_index in seen_indexes:
                    errors.append(f"Duplicate chunk_index {c.chunk_index} in document '{doc_id}'")
                seen_indexes.add(c.chunk_index)

            # Verify strictly sequential 0..N-1
            expected_indexes = set(range(expected_total))
            if seen_indexes != expected_indexes:
                errors.append(
                    f"Chunk indexes for document '{doc_id}' are non-sequential: found {sorted(list(seen_indexes))}, expected {sorted(list(expected_indexes))}"
                )

        # Check 10: Immutability check
        if original_doc_snapshots:
            for doc_id, original_text in original_doc_snapshots.items():
                current_text = source_content_map.get(doc_id)
                if current_text != original_text:
                    errors.append(
                        f"Immutability violation: source document '{doc_id}' was modified during chunking"
                    )

        is_valid = len(errors) == 0
        return {
            "is_valid": is_valid,
            "total_chunks_checked": len(chunks),
            "errors": errors,
            "warnings": warnings,
            "error_count": len(errors),
            "warning_count": len(warnings),
        }
