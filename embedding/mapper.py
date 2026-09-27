from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("embedding.mapper")


class DocumentMapper:
    """Extracts and maps validated documents and chunks into Supabase table records."""

    @classmethod
    def load_validated_documents(cls, validated_path: Path | str) -> Dict[str, Dict[str, Any]]:
        """Load documents dictionary keyed by document_id from validated_documents.json."""
        p = Path(validated_path)
        if not p.exists():
            return {}

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_docs = data.get("documents", [])
        return {doc["document_id"]: doc for doc in raw_docs if "document_id" in doc}

    @classmethod
    def build_document_records(
        cls,
        chunks: List[Dict[str, Any]],
        validated_docs_map: Optional[Dict[str, Dict[str, Any]]] = None,
        allowed_columns: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Build distinct document records for the `documents` table from chunks and validated corpus.
        
        Matches confirmed Supabase schema:
        - document_id
        - title
        - domain
        - storage_path
        - metadata
        """
        docs_by_id: Dict[str, Dict[str, Any]] = {}
        val_map = validated_docs_map or {}

        for chunk in chunks:
            doc_id = chunk["document_id"]
            if doc_id in docs_by_id:
                continue

            val_doc = val_map.get(doc_id, {})

            # Document-level metadata dictionary (preserves all parent attributes)
            metadata = {
                "department": chunk.get("department") or val_doc.get("department"),
                "department_id": chunk.get("department_id") or val_doc.get("department_id"),
                "document_type": chunk.get("document_type") or val_doc.get("document_type"),
                "version": chunk.get("version") or val_doc.get("version"),
                "effective_date": chunk.get("effective_date") or val_doc.get("effective_date"),
                "status": chunk.get("status") or val_doc.get("status", "Active"),
                "audience": chunk.get("audience") or val_doc.get("audience", []),
                "keywords": chunk.get("keywords") or val_doc.get("keywords", []),
                "retrieval_signals": chunk.get("retrieval_signals") or val_doc.get("retrieval_signals", []),
                "systems": chunk.get("systems") or val_doc.get("systems", []),
                "roles": chunk.get("roles") or val_doc.get("roles", []),
                "related_documents": chunk.get("related_documents") or val_doc.get("related_documents", []),
            }

            # Map department to domain at top level
            domain = chunk.get("department") or val_doc.get("department")

            raw_record = {
                "document_id": doc_id,
                "title": chunk.get("title") or val_doc.get("title", doc_id),
                "domain": domain,
                "storage_path": None,  # Preserved as NULL/empty per L0 specification
                "metadata": metadata,
            }

            if allowed_columns:
                # Keep primary key and adapt to specific allowed columns
                record = {k: v for k, v in raw_record.items() if k in allowed_columns}
                record["document_id"] = doc_id
            else:
                record = raw_record

            docs_by_id[doc_id] = record

        return list(docs_by_id.values())

    @staticmethod
    def passage_text(chunk: Dict[str, Any], contextual: bool = True) -> str:
        """Text that is embedded for a chunk.

        With ``contextual=True`` the document title and section heading are
        prepended, so chunks deep inside a document still carry what the
        document is about (e.g. a lab-assignment chunk of a course outline).
        The stored ``chunk_text`` is never changed.
        """
        text = chunk["text"]
        if not contextual:
            return text
        header = "\n".join(p for p in (chunk.get("title"), chunk.get("section")) if p)
        return f"{header}\n\n{text}" if header else text

    @classmethod
    def build_chunk_records(
        cls,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        allowed_columns: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Build chunk records for `document_chunks` table with embedding vectors.
        
        Matches confirmed Supabase schema:
        - chunk_id
        - document_id
        - chunk_index
        - chunk_text
        - section
        - token_count
        - embedding
        - metadata
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
            )

        records: List[Dict[str, Any]] = []

        for chunk, emb in zip(chunks, embeddings):
            chunk_id = chunk["chunk_id"]
            doc_id = chunk["document_id"]
            chunk_text = chunk["text"]

            metadata = {
                "section_number": chunk.get("section_number"),
                "subsection": chunk.get("subsection"),
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
                "character_count": chunk.get("character_count"),
                "total_chunks": chunk.get("total_chunks", 1),
                "department": chunk.get("department"),
                "department_id": chunk.get("department_id"),
                "document_type": chunk.get("document_type"),
                "version": chunk.get("version"),
                "effective_date": chunk.get("effective_date"),
                "audience": chunk.get("audience", []),
                "keywords": chunk.get("keywords", []),
                "retrieval_signals": chunk.get("retrieval_signals", []),
                "systems": chunk.get("systems", []),
                "roles": chunk.get("roles", []),
                "related_documents": chunk.get("related_documents", []),
            }

            raw_record = {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "chunk_index": chunk.get("chunk_index", 0),
                "chunk_text": chunk_text,
                "section": chunk.get("section"),
                "token_count": chunk.get("token_count", 0),
                "embedding": emb,
                "metadata": metadata,
            }

            if allowed_columns:
                record = {k: v for k, v in raw_record.items() if k in allowed_columns}
                record["chunk_id"] = chunk_id
                record["document_id"] = doc_id
                record["embedding"] = emb
                if "chunk_text" in allowed_columns and "chunk_text" not in record:
                    record["chunk_text"] = chunk_text
            else:
                record = raw_record

            records.append(record)

        return records
